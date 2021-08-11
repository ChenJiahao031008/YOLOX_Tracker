#include "YOLOX_Tracker.h"
#include "yolox.h"
#include "objecttracking.h"
#include "DataAssociation.h"

#define INTERVAL 4

int main(int argc, char** argv)
{
    std::string engineFile = "../config/model_trt.engine";
    std::string jsonFile = "../config/config.json";

    YOLOX detector(engineFile);
    ObjectTracking tracker(jsonFile);

    cv::VideoCapture cap = cv::VideoCapture(0);
    double width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    double height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    // cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    // cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    cv::Mat frame;
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

    int count = 1;
    int reDetectedFlag = 0;
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

        // 对当前帧进行预测
        if (reDetectedFlag == 0){
            tracker.RunTracker(frame, previousObjs);
            if (tracker.ImageSimilarityFlag(frame, previousObjs) == 1)
                count = 0;
        }else{
            count = INTERVAL;
        }


        if (count%INTERVAL==0){
            // std::cout << "[INFO] DETECTING..." << std::endl;
            // 当前帧检测
            detector.Detect(frame, currentObjs);
            if ( currentObjs.size() == 0 ){
                std::cout << "[WARNNING] OBJECTS SIZE IS ZERO" << std::endl;
                reDetectedFlag = 1;
                continue;
            }
            // 数据关联处理
            dataAssociation.Association(previousObjs, currentObjs);
            if (currentObjs.size() != 0)
            {
                // 预测：采用当前帧帧初始化
                tracker.InitTracker(frame, currentObjs);
                // 绘制结果
                detector.DrawObjects(frame, currentObjs, "RESULT: SHOW IMAGE");
                previousObjs.clear();
                previousObjs = currentObjs;
                count = 1;
                reDetectedFlag = 0;
            }
            else
            {
                reDetectedFlag = 1;
            }
        }
        else
        {
            // std::cout << "[INFO] TRACKING..." << std::endl;
            // 绘制结果
            detector.DrawObjects(frame, previousObjs, "RESULT: SHOW IMAGE");
            count++;

        }

        // 耗时计算
        auto end = std::chrono::system_clock::now();
        std::cout << "[INFO] Cost Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // debug
        // cv::waitKey(0);

    }
    return 0;
}
