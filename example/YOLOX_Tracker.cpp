#include "YOLOX_Tracker.h"
#include "yolox.h"
#include "objecttracking.h"
#include "DataAssociation.h"

#define INTERVAL 5

int main(int argc, char** argv)
{
    std::string engineFile = "../config/model_trt.engine";
    std::string jsonFile = "../config/config.json";

    YOLOX detector(engineFile);
    ObjectTracking tracker(jsonFile);

    cv::VideoCapture cap = cv::VideoCapture(0);
    double width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    double height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1080);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

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

    int count = 1;
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

        if (count%INTERVAL==0){
            std::cout << "[INFO] DETECTING..." << std::endl;
            // 预测：采用上一帧初始化
            tracker.InitTracker(lastframe, previousObjs);
            // 对当前帧进行预测
            tracker.RunTracker(frame, previousObjs);
            // 当前帧检测
            detector.Detect(frame, currentObjs);
            // 数据关联处理
            dataAssociation.Association(previousObjs, currentObjs);
            // 绘制结果
            detector.DrawObjects(frame, currentObjs, "RESULT: SHOW IMAGE");
            previousObjs = currentObjs;
            count = 1;
            // 预测：采用当前帧帧初始化
            tracker.InitTracker(frame, previousObjs);
        }
        else
        {
            std::cout << "[INFO] TRACKING..." << std::endl;
            // 对当前帧进行预测
            tracker.RunTracker(frame, previousObjs);
            printf("here!\n");
            // 绘制结果
            detector.DrawObjects(frame, previousObjs, "RESULT: SHOW IMAGE");
            lastframe = frame.clone();
            count++;
        }

        // 耗时计算
        auto end = std::chrono::system_clock::now();
        std::cout << "[INFO] Cost Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    }
    return 0;
}
