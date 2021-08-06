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
    double width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    double height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    cv::Mat frame, lastframe;
    std::vector<Object> currentObjs, previousObjs;
    DataAssociation dataAssociation;

    cap.read(frame);
    if (frame.empty())
        std::cout << "[ERRO] Read frame failed!" << std::endl;
    else
        std::cout << "[INFO] Image size : " << frame.size() << std::endl;

    detector.Detect(frame, currentObjs);
    tracker.InitTracker(frame, currentObjs);
    previousObjs = currentObjs;
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
        detector.Detect(frame, currentObjs);
        // 预测：采用上一帧初始化
        tracker.InitTracker(lastframe, previousObjs);
        // 对当前帧进行预测
        tracker.RunTracker(frame, previousObjs);
        detector.DrawObjects(frame, previousObjs, "predictObjs");
        // 数据关联处理
        dataAssociation.Association(previousObjs, currentObjs);
        // 绘图显示
        detector.DrawObjects(frame, currentObjs, "observeObjs");
        // cv::waitKey(0);

        lastframe = frame.clone();
        previousObjs = currentObjs;
        // 耗时计算
        auto end = std::chrono::system_clock::now();
        std::cout << "[INFO] Cost Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    }
    return 0;
}
