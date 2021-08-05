#include "YOLOX_Tracker.h"
#include "yolox.h"
#include "objecttracking.h"
#include "DataAssociation.h"

int main(int argc, char** argv)
{
    std::string engineFile = "../config/model_trt.engine";
    std::string jsonFile = "../config/config.json";

    YOLOX detector(engineFile);
    ObjectTracking tracker(jsonFile);

    cv::VideoCapture cap = cv::VideoCapture(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 848);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame, lastframe;
    std::vector<Object> predictObjs, observeObjs, lastObserveObjs;
    DataAssociation dataAssociation;

    cap.read(frame);
    if (frame.empty())
        std::cout << "[ERRO] Read frame failed!" << std::endl;

    detector.Detect(frame, observeObjs);
    tracker.InitTracker(frame, observeObjs);
    lastObserveObjs = observeObjs;
    lastframe = frame.clone();

    while (cap.isOpened())
    {
        cap.read(frame);
        if (frame.empty())
        {
            std::cout << "[ERRO] Read frame failed!" << std::endl;
            break;
        }
        // detector.Detect(frame);
        auto start = std::chrono::system_clock::now();
        // 当前帧检测
        detector.Detect(frame, observeObjs);
        // 预测：采用上一帧初始化
        tracker.InitTracker(lastframe, lastObserveObjs);
        // 对当前帧进行预测
        tracker.RunTracker(frame, lastObserveObjs);
        predictObjs = lastObserveObjs;
        // detector.DrawObjects(frame, predictObjs, "predictObjs");
        // 数据关联处理
        dataAssociation.Association(predictObjs, observeObjs);
        lastframe = frame.clone();
        lastObserveObjs = observeObjs;
        // 耗时计算
        auto end = std::chrono::system_clock::now();
        // std::cout << "[INFO] Cost Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        // 绘图显示
        detector.DrawObjects(frame, observeObjs, "observeObjs");
        // cv::waitKey(0);
    }
    return 0;
}
