#include "objecttracking.h"

#define USE_OPENMP
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


void ObjectTracking::InitTracker(cv::Mat &frame, std::vector<Object> &vObject)
{
    vTrackers.clear();
    std::vector<Object> newObject;
#ifdef USE_OPENMP
    omp_set_num_threads(10);

#pragma omp parallel for
#endif
    for (size_t i=0; i<vObject.size(); ++i)
    {
        Object obj = vObject[i];
        KCFTracker tracker = KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
        tracker.scale_step = config.scale_step;
        tracker.n_scales = config.num_scales;
        tracker.init(cv::Point2i(obj.rect.tl()), cv::Point2i(obj.rect.br()), frame);
        if (tracker.successflag == 0){
            std::cout << "[WARNNING] NO TRACKER INIT HERE." << std::endl;
            continue;
        }
        vTrackers.push_back(tracker);
        newObject.push_back(obj);
    }
    vObject = newObject;
}

void ObjectTracking::RunTracker(cv::Mat &frame, std::vector<Object> &vObject)
{
    // std::cout << "[DEBUG]  RunTracker " << std::endl;
    assert(vObject.size() == vTrackers.size());

#ifdef USE_OPENMP
    omp_set_num_threads(10);
#pragma omp parallel for
#endif
    for( int i=0; i<vObject.size(); ++i){
        vObject[i].rect = vTrackers[i].update(frame);
    }
     // vTrackers.clear();
}
