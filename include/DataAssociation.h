#include <iostream>

#include "yolox.h"

class DataAssociation
{
private:
    /* data */
public:
    DataAssociation(/* args */);

    ~DataAssociation();

    void InitKalmanFilter();

    void RunKalmanFilter();

    void Association(std::vector<Object> &predictObjs, std::vector<Object> &observeObjs);
};

class CompGreater
{
public:
    // 根据x轴坐标从大到小进行排序
    bool operator()(Object &obj1, Object &obj2)
    {
        return obj1.rect.x > obj2.rect.x;
    }
};
