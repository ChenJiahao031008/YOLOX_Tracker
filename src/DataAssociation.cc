#include "DataAssociation.h"

#define LOST_NFRAMES 1

DataAssociation::DataAssociation()
{
}

DataAssociation::~DataAssociation()
{
}

void DataAssociation::Association(std::vector<Object> &predictObjs, std::vector<Object> &observeObjs)
{
    std::vector<int> vCorresponds(predictObjs.size(), -1);
    std::vector<Object> predictOutputObjs, finialObjs;
    finialObjs = observeObjs;
    predictOutputObjs = predictObjs;
    // std::cout << "[DEDUG] Tracker Num: " << predictObjs.size() << std::endl;

    for (size_t i = 0; i < predictObjs.size(); ++i)
    {
        float maxIoU = 0.0;
        for (size_t j = 0; j < observeObjs.size(); ++j)
        {
            cv::Rect rectIntersection = (predictObjs[i].rect) & (observeObjs[j].rect);
            cv::Rect rectUnion = (predictObjs[i].rect) | (observeObjs[j].rect);
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
            std::cout << "[DEDUG] Association: " << i << "; " << vCorresponds[i] << std::endl;
            predictOutputObjs[i].nFrames++;
            predictOutputObjs[i].lostFrames = 0;
            predictOutputObjs[i].label = finialObjs[vCorresponds[i]].label;
            predictOutputObjs[i].prob = finialObjs[vCorresponds[i]].prob;
            // TODO: 补充数据融合部分(采用卡尔曼滤波)
            finialObjs[vCorresponds[i]] = predictOutputObjs[i];
        }
        else
        {
            // 只有预测存在
            predictOutputObjs[i].lostFrames++;
            if (predictObjs[i].prob > 0.75){
                if (predictObjs[i].lostFrames < LOST_NFRAMES && predictObjs[i].nFrames > 1)
                    finialObjs.push_back(predictOutputObjs[i]);
                else
                    predictOutputObjs[i].nFrames = 0;
            }else{
                if (predictObjs[i].lostFrames < 2 * LOST_NFRAMES && predictObjs[i].nFrames > 1)
                    finialObjs.push_back(predictOutputObjs[i]);
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
