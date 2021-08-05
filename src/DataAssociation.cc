#include "DataAssociation.h"

DataAssociation::DataAssociation()
{
}

DataAssociation::~DataAssociation()
{
}

void DataAssociation::Association(std::vector<Object> &predictObjs, std::vector<Object> &observeObjs)
{
    // sort(predictObjs.begin(), predictObjs.end(), CompGreater());
    // sort(observeObjs.begin(), observeObjs.end(), CompGreater());

    std::vector<int> vCorresponds(predictObjs.size(), -1);
    std::vector<Object> predictOutputObjs, finialObjs;
    finialObjs = observeObjs;
    predictOutputObjs = predictObjs;
    // std::cout << "predictOutputObjs.size() : " << predictOutputObjs.size() << std::endl;

    // std::cout << "Association..." << std::endl;
    for (size_t i = 0; i < predictObjs.size(); ++i)
    {
        float maxIoU = 0.0;
        // std::cout << "TargeLabel : " << predictObjs[i].label << std::endl;
        for (size_t j = 0; j < observeObjs.size(); ++j)
        {
            // cv::Rect rectIntersection(0, 0, 0, 0);
            // cv::Rect rectUnion(0, 0, 0, 0);
            cv::Rect rectIntersection = (predictObjs[i].rect) & (observeObjs[j].rect);
            cv::Rect rectUnion = (predictObjs[i].rect) | (observeObjs[j].rect);
            float currentIoU = rectIntersection.area()*1.0 / rectUnion.area();
            // std::cout << "currentIoU : " << currentIoU << std::endl;
            // std::cout << "currentLabel : " << observeObjs[j].label << std::endl;
            // if (currentIoU < 0.8)
            //     continue;
            if (currentIoU < 0.65 )
                continue;
            if (currentIoU > maxIoU){
                maxIoU = currentIoU;
                vCorresponds[i] = j;
            }
        }
        // 预测和观测都存在
        if (vCorresponds[i] != -1)
        {
            // std::cout << "j : " << vCorresponds[i] << std::endl;
            predictOutputObjs[i].nFrames++;
            predictOutputObjs[i].lostFrames = 0;
            // TODO: 补充数据融合部分(采用卡尔曼滤波)
            finialObjs[vCorresponds[i]] = predictObjs[i];
        }
        else
        {
            // 只有预测存在
            size_t j = vCorresponds[i];
            predictOutputObjs[i].nFrames = 0;
            predictOutputObjs[i].lostFrames++;
            if (predictObjs[i].lostFrames < 10 || predictObjs[i].nFrames > 3)
            {
                finialObjs.push_back(predictOutputObjs[i]);
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
