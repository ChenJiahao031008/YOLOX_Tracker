#include "DataAssociation.h"

#define LOST_NFRAMES 2

DataAssociation::DataAssociation()
{
}

DataAssociation::~DataAssociation()
{
}

void DataAssociation::Association(std::vector<Object> &predictObjs, std::vector<Object> &observeObjs)
{
    std::vector<int> vCorresponds(predictObjs.size(), -1);
    std::vector<Object> predictOutputObjs, predictinputObjs, finialObjs;
    finialObjs = observeObjs;

    for (auto &obj: predictObjs){
        if (obj.similarity != -1){
            predictinputObjs.push_back(obj);
        }
    }
    predictinputObjs = predictOutputObjs;

    if (predictinputObjs.size()==0){
        std::cout << "[WARNNING] EMPTY SIZE! " << std::endl;
        predictObjs = predictinputObjs;
        observeObjs = finialObjs;
        return;
    }

    for (size_t i = 0; i < predictinputObjs.size(); ++i)
    {
        float maxIoU = 0.0;
        for (size_t j = 0; j < observeObjs.size(); ++j)
        {
            cv::Rect rectIntersection = (predictinputObjs[i].rect) & (observeObjs[j].rect);
            cv::Rect rectUnion = (predictinputObjs[i].rect) | (observeObjs[j].rect);
            float currentIoU = rectIntersection.area()*1.0 / rectUnion.area();
            if (currentIoU < 0.5 )
                continue;
            if (currentIoU > maxIoU){
                maxIoU = currentIoU;
                vCorresponds[i] = j;
            }
        }

        // 预测和观测都存在
        if (vCorresponds[i] != -1)
        {
            predictOutputObjs[i].nFrames++;
            predictOutputObjs[i].lostFrames = 0;
            // 更新标签
            predictOutputObjs[i].label = finialObjs[vCorresponds[i]].label;
            // 计算融合后目标框大小
            // 计算权重
            float weight = (predictOutputObjs[i].prob)  *1.0 / (predictOutputObjs[i].prob + finialObjs[vCorresponds[i]].prob);
            cv::Rect2f fuse;
            // 根据权重分配目标框的大小
            fuse.width = weight * predictOutputObjs[i].rect.width + (1 - weight) * finialObjs[vCorresponds[i]].rect.width;
            fuse.height = weight * predictOutputObjs[i].rect.height + (1 - weight) * finialObjs[vCorresponds[i]].rect.height;
            // 修正目标框的位置
            fuse.x = weight * predictOutputObjs[i].rect.x + (1 - weight) * finialObjs[vCorresponds[i]].rect.x;
            fuse.y = weight * predictOutputObjs[i].rect.y + (1 - weight) * finialObjs[vCorresponds[i]].rect.y;
            // 更新目标框
            predictOutputObjs[i].rect = fuse;
            // 更新概率
            predictOutputObjs[i].prob = finialObjs[vCorresponds[i]].prob;
            // TODO: 补充数据融合部分(采用卡尔曼滤波)
            finialObjs[vCorresponds[i]] = predictOutputObjs[i];
        }
        else
        {
            // 只有预测存在
            predictOutputObjs[i].lostFrames++;
            if (predictinputObjs[i].prob < 0) continue;
            if (predictinputObjs[i].prob > 0.75){
                if (predictinputObjs[i].lostFrames < (LOST_NFRAMES+1) && predictinputObjs[i].nFrames > 3){
                    predictOutputObjs[i].prob -= 0.16;
                    finialObjs.push_back(predictOutputObjs[i]);
                }
                else
                    predictOutputObjs[i].nFrames = 0;
            }else{
                if (predictinputObjs[i].lostFrames < 1.5 * (LOST_NFRAMES+1) && predictinputObjs[i].nFrames > 3){
                    predictOutputObjs[i].prob -= 0.08;
                    finialObjs.push_back(predictOutputObjs[i]);
                }
                else
                    predictOutputObjs[i].nFrames = 0;
            }
        }
    }
    predictObjs = predictOutputObjs;
    observeObjs = finialObjs;
}

void DataAssociation::RunKalmanFilter()
{
}

void DataAssociation::InitKalmanFilter()
{
}
