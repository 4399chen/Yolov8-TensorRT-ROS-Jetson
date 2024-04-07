#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <ros/ros.h>

#include "image_transport/image_transport.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CompressedImage.h"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "std_msgs/Float32.h"

#include <vision_msgs/BoundingBox2D.h>
#include <vision_msgs/BoundingBox2DArray.h>
#include <vision_msgs/BoundingBox3D.h>
#include <vision_msgs/BoundingBox3DArray.h>
#include <vision_msgs/Classification2D.h>
#include <vision_msgs/Classification3D.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/Detection3DArray.h>
#include <vision_msgs/Detection3D.h>
#include <vision_msgs/ObjectHypothesis.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>
#include <vision_msgs/VisionInfo.h>

const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

ros::Publisher result_pub;

void Callback(const sensor_msgs::ImageConstPtr &msg, YOLOv8* yolov8, ros::NodeHandle& nh){
    result_pub = nh.advertise<vision_msgs::Detection2DArray>("/boundingboxes", 1);

    //make a message to save
    vision_msgs::Detection2DArray detect_array_msg;

    // detect_array_msg.image_header = msg->header;

    cv::Mat             res, img;
    cv::Size            size = cv::Size{640, 640};
    std::vector<Object> objs;

    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    img = cv_ptr->image;
    
    //clear last answer
    objs.clear();

    yolov8->copy_from_Mat(img, size);

    //get the detect start time
    auto start = std::chrono::system_clock::now();

    yolov8->infer();

    //get the detect end time
    auto end = std::chrono::system_clock::now();

    yolov8->postprocess(objs);

    //draw the bboxs on images
    yolov8->draw_objects(img, res, objs, CLASS_NAMES, COLORS);
    
    //get the detect time
    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    
    printf("###\n");

    //print detect time
    printf("cost %.4lf ms\n", tc);

    for (int i=0; i<objs.size(); i++) {
        vision_msgs::Detection2D detect_msg;
        vision_msgs::ObjectHypothesisWithPose hypo;

        hypo.id = objs[i].label;  // 设置类别ID
        hypo.score = objs[i].prob;  // 设置置信度
        detect_msg.results.push_back(hypo);

        detect_msg.bbox.center.x = objs[i].rect.x + objs[i].rect.width / 2;
        detect_msg.bbox.center.y = objs[i].rect.y + objs[i].rect.height / 2;
        detect_msg.bbox.size_x = objs[i].rect.width;
        detect_msg.bbox.size_y = objs[i].rect.height;

        detect_array_msg.detections.push_back(detect_msg);
    }

    result_pub.publish(detect_array_msg);

    //show the video with boundingboxs
    cv::imshow("result_img", res);
    cv::waitKey(10);
}


int main(int argc, char** argv)
{
    const std::string engine_file_path{argv[1]};
    const std::string path{argv[2]};

    std::vector<std::string> imagePathList;
    bool                     isVideo{false};
    bool                     isROSTopic{false};

    assert(argc == 3);

    auto yolov8 = new YOLOv8(engine_file_path);

    std::string input_image_topic_name = path;

    yolov8->make_pipe(true);

    // cv::namedWindow("result_img", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("result_img", cv::WINDOW_KEEPRATIO);

    //make a ROS node "yolov8_node"
    ros::init(argc, argv, "yolov8_node");

    //make a handle
    ros::NodeHandle nh;

    result_pub = nh.advertise<vision_msgs::Detection2DArray>("/boundingboxes", 1);

    if (IsFile(path)) {
        std::string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png") {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov"
                 || suffix == "mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (IsFolder(path)) {
        cv::glob(path + "/*.jpg", imagePathList);
    }
    else if (path.at(0) == '/'){
        isROSTopic = true;
    }    

    if (isROSTopic) {
        printf("Is ROS Topic.\n");

        image_transport::ImageTransport it(nh);

        image_transport::Subscriber sub = it.subscribe(input_image_topic_name, 1, boost::bind(&Callback, _1, yolov8, nh) );

        ros::spin();
    }
    else {
        vision_msgs::Detection2DArray detect_array_msg;
        cv::Mat             res, image;
        cv::Size            size = cv::Size{640, 640};
        std::vector<Object> objs;

        if (isVideo) {
            cv::VideoCapture cap(path);

            if (!cap.isOpened()) {
                printf("can not open %s\n", path.c_str());
                return -1;
            }
            while (cap.read(image)) {
                //clear last answer
                objs.clear();

                yolov8->copy_from_Mat(image, size);

                //get the detect start time
                auto start = std::chrono::system_clock::now();

                yolov8->infer();

                //get the detect end time
                auto end = std::chrono::system_clock::now();

                yolov8->postprocess(objs);

                //draw the bboxs on images
                yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
                
                //get the detect time
                auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
                
                printf("####################################################\n");

                //print detect time
                printf("cost %.4lf ms\n", tc);

                for (int i=0; i<objs.size(); i++) {
                    vision_msgs::Detection2D detect_msg;
                    vision_msgs::ObjectHypothesisWithPose hypo;

                    hypo.id = objs[i].label;  // 设置类别ID
                    hypo.score = objs[i].prob;  // 设置置信度
                    detect_msg.results.push_back(hypo);

                    detect_msg.bbox.center.x = objs[i].rect.x + objs[i].rect.width / 2;
                    detect_msg.bbox.center.y = objs[i].rect.y + objs[i].rect.height / 2;
                    detect_msg.bbox.size_x = objs[i].rect.width;
                    detect_msg.bbox.size_y = objs[i].rect.height;

                    detect_array_msg.detections.push_back(detect_msg);
                }
                result_pub.publish(detect_array_msg);
                //show the video with boundingboxs
                cv::imshow("result_img", res);

                //pull "Q" to exit
                if (cv::waitKey(10) == 'q') {
                    break;
                }
            }
        }
        else {
            for (auto& path : imagePathList) {
                objs.clear();
                image = cv::imread(path);
                yolov8->copy_from_Mat(image, size);
                auto start = std::chrono::system_clock::now();
                yolov8->infer();
                auto end = std::chrono::system_clock::now();
                yolov8->postprocess(objs);
                yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
                auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
                printf("####################################################\n");
                printf("cost %2.4lf ms\n", tc);
                for (int i=0; i<objs.size(); i++) {
                    vision_msgs::Detection2D detect_msg;
                    vision_msgs::ObjectHypothesisWithPose hypo;

                    hypo.id = objs[i].label;  // 设置类别ID
                    hypo.score = objs[i].prob;  // 设置置信度
                    detect_msg.results.push_back(hypo);

                    detect_msg.bbox.center.x = objs[i].rect.x + objs[i].rect.width / 2;
                    detect_msg.bbox.center.y = objs[i].rect.y + objs[i].rect.height / 2;
                    detect_msg.bbox.size_x = objs[i].rect.width;
                    detect_msg.bbox.size_y = objs[i].rect.height;

                    detect_array_msg.detections.push_back(detect_msg);
                }
                result_pub.publish(detect_array_msg);
                cv::imshow("result_img", res);
                cv::waitKey(0);
            }
        }
    }

    cv::destroyAllWindows();

    delete yolov8;

    return 0;
}
