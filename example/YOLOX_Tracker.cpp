#include "YOLOX_Tracker.h"
#include "yolox.h"
#include "objecttracking.h"

int main(int argc, char** argv)
{
    std::string engineFile = "../config/model_trt.engine";
    std::string jsonFile = "../config/config.json";

    YOLOX detector(engineFile);
    ObjectTracking tracker(jsonFile);

    cv::VideoCapture cap = cv::VideoCapture(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 848);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;
    std::vector<Object> objects;

    cap.read(frame);
    if (frame.empty())
        std::cout << "[ERRO] Read frame failed!" << std::endl;

    detector.Detect(frame, objects);
    tracker.InitTrackerOnce(frame, objects);

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
        auto tmp = objects;
        detector.Detect(frame, tmp);
        tracker.RunTracker(frame, objects);
        detector.DrawObjects(frame, objects);
        auto end = std::chrono::system_clock::now();
        std::cout << "[INFO] Cost Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    return 0;
}
