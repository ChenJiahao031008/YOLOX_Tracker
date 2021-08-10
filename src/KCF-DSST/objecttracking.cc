#include "objecttracking.h"

#define USE_OPENMP
#define METHOD 3

#ifdef USE_OPENMP
#include "omp.h"
#endif

ObjectTracking::ObjectTracking(const std::string &jsonFile)
{

#ifdef _OPENMP
    std::cout << "[INFO] START OPENMP!" << std::endl;
#endif

    if (parse_config(jsonFile, config))
    {
        HOG = config.hog;
        LAB = config.lab;
        FIXEDWINDOW = config.fixed_window;
        SILENT = config.silent;
        MULTISCALE = config.multi_scale;

        std::cout << "[INFO]\n\tHOG = " << HOG << std::endl;
        std::cout << "\tLAB = " << LAB << std::endl;
        std::cout << "\tFIXEDWINDOW = " << FIXEDWINDOW << std::endl;
        std::cout << "\tSILENT = " << SILENT << std::endl;
        std::cout << "\tMULTISCALE = " << MULTISCALE << std::endl;

        std::cout << "\tSCALE STEP = " << config.scale_step << std::endl;
        std::cout << "\tNUM SCALES = " << config.num_scales << std::endl;
    }
    else
    {
        std::cout << "[ERRO] CONFIG ERROR!" << std::endl;
    }

}

ObjectTracking::~ObjectTracking()
{
    vTrackers.clear();
}

bool ObjectTracking::parse_config(const std::string &path, sys_config &config)
{
    std::ifstream ifs;
    Json::Reader reader;
    Json::Value root;

    ifs.open(path, std::ios::in | std::ios::binary);
    if (ifs.is_open() == false)
    {
        std::cout << "Open file failed!\n";
        return false;
    }

    if (!reader.parse(ifs, root, false))
    {
        std::cout << "Read file failed!\n";
        ifs.close();
        return false;
    }

    memset(&config, 0, sizeof(sys_config));

    config.hog = root["hog"].asInt();
    config.lab = root["lab"].asInt();

    config.fixed_window = root["fixed window"].asInt();
    config.multi_scale = root["multi scale"].asInt();

    config.silent = root["silent"].asInt();

    config.scale_step = root["scale step"].asFloat();
    config.num_scales = root["num scales"].asInt();

    ifs.close();
    return true;
}


void ObjectTracking::InitTracker(cv::Mat &image, std::vector<Object> &vObject)
{
    // std::cout << "[DEBUG] InitTracker " << std::endl;

    std::vector<cv::Mat> frames(vObject.size());
    std::vector<int> successFlags(vObject.size(),1);

    vTrackers.clear();
    vTrackers.resize(vObject.size());
    keyFrame = image.clone();

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (size_t i=0; i<vObject.size(); ++i){
        cv::Mat frame = image.clone();
        frames[i] = frame;
    }

#ifdef USE_OPENMP
    omp_set_num_threads(12);
#pragma omp parallel for
#else
        cv::Mat frame = image;
#endif
    for (size_t i=0; i<vObject.size(); ++i)
    {
        Object obj = vObject[i];
        KCFTracker tracker = KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
        tracker.scale_step = config.scale_step;
        tracker.n_scales = config.num_scales;
        tracker.init(cv::Point2i(obj.rect.tl()), cv::Point2i(obj.rect.br()), frames[i]);
        if (tracker.successflag == 0){
            std::cout << "[WARNNING] NO TRACKER INIT SUCCESS." << std::endl;
            successFlags[i] = 0;
        }else{
            successFlags[i] = 1;
        }
        vTrackers[i] = tracker;
    }

    std::vector<Object> newObject;
    std::vector<KCFTracker> newTrackers;
    for (size_t i=0; i<vObject.size(); ++i){
        if (successFlags[i] == 0)
            continue;
        newTrackers.emplace_back(vTrackers[i]);
        newObject.emplace_back(vObject[i]);
    }
    vTrackers = newTrackers;
    vObject = newObject;
}

void ObjectTracking::RunTracker(cv::Mat &image, std::vector<Object> &vObject)
{
    // std::cout << "[DEBUG]  RunTracker " << std::endl;
    assert(vObject.size() == vTrackers.size());
    std::vector<cv::Mat> frames(vObject.size());

#ifdef USE_OPENMP
    omp_set_num_threads(12);
#pragma omp parallel for
#endif
    for (size_t i=0; i<vObject.size(); ++i){
        cv::Mat frame = image.clone();
        frames[i] = frame;
    }
#ifdef USE_OPENMP
#pragma omp parallel for
#else
        cv::Mat frame = image;
#endif
    for( int i=0; i<vObject.size(); ++i){
        vObject[i].rect = vTrackers[i].update(frames[i]);
        vObject[i].nFrames++;
    }
    // vTrackers.clear();
}

int ObjectTracking::ImageSimilarityFlag(const cv::Mat &Frame, std::vector<Object> &vObject)
{
    // auto start = std::chrono::system_clock::now();
    if (vObject.size()==0)
        return 1;
    int detectedFlag = 0;
    std::vector<int> flags(vObject.size());

    std::vector<cv::Mat> currentFrames(vObject.size());
    std::vector<cv::Mat> keyFrameCopys(vObject.size());

#ifdef USE_OPENMP
    omp_set_num_threads(12);
#pragma omp parallel for
#endif
    for (size_t i=0; i<vObject.size();++i){
        cv::Mat currentFrame = Frame.clone();
        cv::Mat keyFrameCopy = keyFrame.clone();
        currentFrames[i] = currentFrame;
        keyFrameCopys[i] = keyFrameCopy;
    }
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (size_t i=0; i<vObject.size(); ++i ){
        cv::Rect2i rectangle = cv::Rect((int)vObject[i].rect.x, (int)vObject[i].rect.y, (int)vObject[i].rect.width, (int)vObject[i].rect.height);

        rectangle &= cv::Rect(0, 0, currentFrames[i].cols, currentFrames[i].rows);

        cv::Mat roi1 = currentFrames[i](rectangle);
        cv::Mat roi2 = keyFrameCopys[i](rectangle);

        cv::Mat hist1, hist2;
        Histogram(roi1, hist1);
        Histogram(roi2, hist2);

        int flag = CompareHist(hist1, hist2);
        vObject[i].similarity = flag;
        flags[i] = flag;
    }

    // auto end = std::chrono::system_clock::now();
    // std::cout << "[INFO] CV_COMP_BHATTACHARYYA Cost Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    for (auto &flag: flags){
        if (flag == -1)
        {
            detectedFlag = 1;
            break;
        }
    }

    // if (detectedFlag == 1){
    //     std::cout << "[INFO] DETECTOR START " << std::endl;
    // }else{
    //     std::cout << "[INFO] TRACKING START " << std::endl;
    // }
    return detectedFlag;
}

void ObjectTracking::Histogram(const cv::Mat &image, cv::Mat &hist){
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    //需要计算的图像的通道，灰度图像为0，BGR图像需要指定B,G,R
    const int channels[] = {0};
    //设置直方图维度
    int dims = 1;
    //直方图每一个维度划分的柱条的数目
    const int histSize[] = {256};
    //取值区间
    float pranges[] = {0, 255};
    const float *ranges[] = {pranges};
    //计算直方图
    cv::calcHist(&gray, 1, channels, cv::Mat(), hist, dims, histSize, ranges, true, false);
}

int ObjectTracking::CompareHist(const cv::Mat &hist1, const cv::Mat &hist2)
{
    int flag = -1;
    cv::normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    // 有四种方法可以选择：相关性比较，卡方检查，十字交叉性，巴氏距离
    int methods[4] = {CV_COMP_CORREL, CV_COMP_CHISQR, CV_COMP_INTERSECT, CV_COMP_BHATTACHARYYA};
    // 巴氏距离检测精度高但是速度稍慢
    double score = cv::compareHist(hist1, hist2, methods[METHOD]);
    // std::cout << "[DEBUG] Similarity Score: " << score << std::endl;

    // TODO: 除了最后一个巴氏距离，其他参数都没有试验过
    if (METHOD == 0 ){
        if (score > 0.70)
            flag = 1;
    }else if (METHOD == 1){
        if (score < 60.0)
            flag = 1;
    }else if (METHOD == 2 ){
        if (score > 25.0)
            flag = 1;
    }else if (METHOD == 3 ){
        if (score < 0.20)
            flag = 1;
    }else{
            std::cout  << "[ERRO] PLEASE CHECK METHOD !" << std::endl;
    }

    // std::cout << "[INFO] Similarity flag: " << flag << std::endl;
    return flag;
}


