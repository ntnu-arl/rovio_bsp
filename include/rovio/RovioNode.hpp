/*
* Copyright (c) 2014, Autonomous Systems Lab
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * Neither the name of the Autonomous Systems Lab, ETH Zurich nor the
* names of its contributors may be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*/

/*
 * This code has been modified from the original to implement the propagation
 * steps and data transactions described as part of the Uncertainty-aware Receding
 * Horizon Exploration and Mapping planner.
 * 
 * Authors of modifications: 2017, Christos Papachristos, University of Nevada, Reno
 */

#ifndef ROVIO_ROVIONODE_HPP_
#define ROVIO_ROVIONODE_HPP_

#include <memory>
#include <mutex>
#include <queue>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <image_transport/image_transport.h>

#include <rovio/SrvResetToPose.h>
#include "rovio/RovioFilter.hpp"
#include "rovio/CoordinateTransform/RovioOutput.hpp"
#include "rovio/CoordinateTransform/FeatureOutput.hpp"
#include "rovio/CoordinateTransform/FeatureOutputReadable.hpp"
#include "rovio/CoordinateTransform/YprOutput.hpp"
#include "rovio/CoordinateTransform/LandmarkOutput.hpp"

// Bsp: extension
#include <tf/transform_listener.h>
// Bsp: visualization extras
#include <visualization_msgs/MarkerArray.h>

namespace rovio
{

/** \brief Class, defining the Rovio Node
 *
 *  @tparam FILTER  - \ref rovio::RovioFilter
 */
template <typename FILTER>
class RovioNode
{
public:
  // Filter Stuff
  typedef FILTER mtFilter;
  std::shared_ptr<mtFilter> mpFilter_;
  typedef typename mtFilter::mtFilterState mtFilterState;
  typedef typename mtFilterState::mtState mtState;
  typedef typename mtFilter::mtPrediction::mtMeas mtPredictionMeas;
  mtPredictionMeas predictionMeas_;
  typedef typename std::tuple_element<0, typename mtFilter::mtUpdates>::type mtImgUpdate;
  typedef typename mtImgUpdate::mtMeas mtImgMeas;
  mtImgMeas imgUpdateMeas_;
  mtImgUpdate *mpImgUpdate_;
  typedef typename std::tuple_element<1, typename mtFilter::mtUpdates>::type mtPoseUpdate;
  typedef typename mtPoseUpdate::mtMeas mtPoseMeas;
  mtPoseMeas poseUpdateMeas_;
  mtPoseUpdate *mpPoseUpdate_;

  // Bsp: Filter extras
  typedef typename mtFilter::mt_BspFeatureParams mt_BspFeatureParams;

  struct FilterInitializationState
  {
    FilterInitializationState()
        : WrWM_(V3D::Zero()),
          state_(State::WaitForInitUsingAccel) {}

    enum class State
    {
      // Initialize the filter using accelerometer measurement on the next
      // opportunity.
      WaitForInitUsingAccel,
      // Initialize the filter using an external pose on the next opportunity.
      WaitForInitExternalPose,
      // The filter is initialized.
      Initialized
    } state_;

    // Buffer to hold the initial pose that should be set during initialization
    // with the state WaitForInitExternalPose.
    V3D WrWM_;
    QPD qMW_;

    explicit operator bool() const
    {
      return isInitialized();
    }

    bool isInitialized() const
    {
      return (state_ == State::Initialized);
    }
  };
  FilterInitializationState init_state_;

  bool forceOdometryPublishing_;
  bool forceTransformPublishing_;
  bool forceExtrinsicsPublishing_;
  bool forceImuBiasPublishing_;
  bool forcePclPublishing_;
  bool forceMarkersPublishing_;
  bool forcePatchPublishing_;
  bool gotFirstMessages_;
  std::mutex m_filter_;

  // Nodes, Subscriber, Publishers
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  image_transport::ImageTransport it_;
  ros::Subscriber subImu_;
  ros::Subscriber subImg0_;
  ros::Subscriber subImg1_;
  ros::Subscriber subImg2_; //CUSTOMZIATION
  ros::Subscriber subGroundtruth_;
  ros::ServiceServer srvResetFilter_;
  ros::ServiceServer srvResetToPoseFilter_;
  ros::Publisher pubOdometry_;
  ros::Publisher pubTransform_;
  tf::TransformBroadcaster tb_;
  ros::Publisher pubPcl_;     /**<Publisher: Ros point cloud, visualizing the landmarks.*/
  ros::Publisher pubPatch_;   /**<Publisher: Patch data.*/
  ros::Publisher pubMarkers_; /**<Publisher: Ros line marker, indicating the depth uncertainty of a landmark.*/
  ros::Publisher pubExtrinsics_[mtState::nCam_];
  ros::Publisher pubImuBias_;
  ros::Publisher pubDecimated_;
  image_transport::Publisher image_pub_;
  

  // Ros Messages
  geometry_msgs::TransformStamped transformMsg_;
  nav_msgs::Odometry odometryMsg_;
  geometry_msgs::PoseWithCovarianceStamped extrinsicsMsg_[mtState::nCam_];
  sensor_msgs::PointCloud2 pclMsg_;
  sensor_msgs::PointCloud2 patchMsg_;
  visualization_msgs::Marker markerMsg_;
  sensor_msgs::Imu imuBiasMsg_;
  sensor_msgs::Image decimatedImage_;
  int msgSeq_;

  // Rovio outputs and coordinate transformations
  typedef StandardOutput mtOutput;
  mtOutput cameraOutput_;
  MXD cameraOutputCov_;
  mtOutput imuOutput_;
  MXD imuOutputCov_;
  CameraOutputCT<mtState> cameraOutputCT_;
  ImuOutputCT<mtState> imuOutputCT_;
  rovio::TransformFeatureOutputCT<mtState> transformFeatureOutputCT_;
  rovio::LandmarkOutputImuCT<mtState> landmarkOutputImuCT_;
  rovio::FeatureOutput featureOutput_;
  rovio::LandmarkOutput landmarkOutput_;
  MXD featureOutputCov_;
  MXD landmarkOutputCov_;
  rovio::FeatureOutputReadableCT featureOutputReadableCT_;
  rovio::FeatureOutputReadable featureOutputReadable_;
  MXD featureOutputReadableCov_;

  // ROS names for output tf frames.
  std::string map_frame_;
  std::string world_frame_;
  std::string camera_frame_;
  std::string imu_frame_;

  //CUSTOMIZATION
  double imu_offset_ = 0.0;                     //time offset added to IMU messages to sync with camera images
  double cam0_offset_ = 0.0;                    //time offset added to camera 0 messages to sync with IMU images
  double cam1_offset_ = 0.0;                    //time offset added to camera 1 messages to sync with IMU images
  double cam2_offset_ = 0.0;                    //time offset added to camera 2 messages to sync with IMU images
  bool resize_input_image_ = false;             //should input images be resized? typically for larger input images to be scaled down - CAMERA INTRINSICS NOT SCALED ACCORDINGLY CHECK CALIBRATION FILE
  double resize_factor_ = 0.5;                  //factor by which input images should be rescaled cannot be greater than 1.0 - CAMERA INTRINSICS NOT SCALED ACCORDINGLY CHECK CALIBRATION FILE
  bool histogram_equalize_8bit_images_ = false; //use CLAHE to histogram equalize input intensity images, only for 8-bit images. 16-bit image equalization for feature detection does not require this to be on.
  cv::Ptr<cv::CLAHE> clahe;
  double clahe_clip_limit_ = 4.0;               //number of pixels used to clip the CDF for histogram equalization
  double clahe_grid_size_ = 8;                  //clahe_grid_size_ x clahe_grid_size_ pixel neighborhood used 
  //CUSTOMIZATION

  // Bsp: node variables
  ros::Time bsp_rootmap_stamp_;
  uint32_t bsp_planning_seq_;
  tf::TransformListener bsp_tl_;
  tf::Vector3 bsp_T_;
  tf::Quaternion bsp_Q_;
  // Bsp: pack and send belief (filter) state
  ros::ServiceServer BSP_servFilterState_;
  // Bsp: get and propagate, then pack and send belief (filter) state
  ros::ServiceServer BSP_servPropagateFilterState_;
  // Bsp: visualization extras
  ros::Publisher BSP_pubFrustum_;
  visualization_msgs::Marker BSP_frustumMsg_;
  ros::Publisher BSP_pubBearingArrows_;
  visualization_msgs::MarkerArray BSP_bearingArrowArrayMsg_;

  /** \brief Constructor
   */
  RovioNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private, std::shared_ptr<mtFilter> mpFilter)
      : nh_(nh), nh_private_(nh_private), it_(nh_private_), mpFilter_(mpFilter), transformFeatureOutputCT_(&mpFilter->multiCamera_), landmarkOutputImuCT_(&mpFilter->multiCamera_),
        cameraOutputCov_((int)(mtOutput::D_), (int)(mtOutput::D_)), featureOutputCov_((int)(FeatureOutput::D_), (int)(FeatureOutput::D_)), landmarkOutputCov_(3, 3),
        featureOutputReadableCov_((int)(FeatureOutputReadable::D_), (int)(FeatureOutputReadable::D_))
  {
#ifndef NDEBUG
    ROS_WARN("====================== Debug Mode ======================");
#endif
    mpImgUpdate_ = &std::get<0>(mpFilter_->mUpdates_);
    mpPoseUpdate_ = &std::get<1>(mpFilter_->mUpdates_);
    forceOdometryPublishing_ = false;
    forceTransformPublishing_ = false;
    forceExtrinsicsPublishing_ = false;
    forceImuBiasPublishing_ = false;
    forcePclPublishing_ = false;
    forceMarkersPublishing_ = false;
    forcePatchPublishing_ = false;
    gotFirstMessages_ = false;

    /*
     * Bsp: switch subscriptions, advertising, services
     */
    if (mpFilter_->bspFilter_)
    {
      // Advertise topics
      BSP_pubFrustum_ = nh_.advertise<visualization_msgs::Marker>("bsp_rovio/frustum", 1);
      BSP_pubBearingArrows_ = nh_.advertise<visualization_msgs::MarkerArray>("bsp_rovio/bearing_arrows", 1);

      // Bsp: get and propagate, then pack and send belief (filter) state
      BSP_servPropagateFilterState_ = nh_.advertiseService("bsp_rovio/propagate_filter_state", &RovioNode::BSP_servPropagateFilterStateCallback, this);

      bsp_rootmap_stamp_ = ros::Time(0);
      bsp_planning_seq_ = 0;
    }
    else
    {
      // Subscribe topics
      subImu_ = nh_.subscribe("imu0", 1000, &RovioNode::imuCallback, this);
      subImg0_ = nh_.subscribe("cam0/image_raw", 1000, &RovioNode::imgCallback0, this);
      subImg1_ = nh_.subscribe("cam1/image_raw", 1000, &RovioNode::imgCallback1, this);
      subImg2_ = nh_.subscribe("cam2/image_raw", 1000, &RovioNode::imgCallback2, this); //CUSTOMIZATION
      subGroundtruth_ = nh_.subscribe("pose", 1000, &RovioNode::groundtruthCallback, this);

      // Initialize ROS service servers.
      srvResetFilter_ = nh_.advertiseService("rovio/reset", &RovioNode::resetServiceCallback, this);
      srvResetToPoseFilter_ = nh_.advertiseService("rovio/reset_to_pose", &RovioNode::resetToPoseServiceCallback, this);

      // Advertise topics
      pubTransform_ = nh_.advertise<geometry_msgs::TransformStamped>("rovio/transform", 1);
      pubOdometry_ = nh_.advertise<nav_msgs::Odometry>("rovio/odometry", 1);
      pubPcl_ = nh_.advertise<sensor_msgs::PointCloud2>("rovio/pcl", 1);
      pubPatch_ = nh_.advertise<sensor_msgs::PointCloud2>("rovio/patch", 1);
      pubMarkers_ = nh_.advertise<visualization_msgs::Marker>("rovio/markers", 1);
      pubDecimated_ = nh_.advertise<sensor_msgs::Image>("cam0/image_decimated", 1);
      image_pub_ = it_.advertise("feature_tracking_image", 1);
      
      for (int camID = 0; camID < mtState::nCam_; camID++)
      {
        pubExtrinsics_[camID] = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("rovio/extrinsics" + std::to_string(camID), 1);
      }
      pubImuBias_ = nh_.advertise<sensor_msgs::Imu>("rovio/imu_biases", 1);

      // Bsp: pack and send belief (filter) state
      BSP_servFilterState_ = nh_.advertiseService("rovio/send_filter_state", &RovioNode::BSP_servFilterStateCallback, this);
      // Bsp: visualization extras
      BSP_pubFrustum_ = nh_.advertise<visualization_msgs::Marker>("rovio/frustum", 1);
      BSP_pubBearingArrows_ = nh_.advertise<visualization_msgs::MarkerArray>("rovio/bearing_arrows", 1);
    }

    // Handle coordinate frame naming
    map_frame_ = "map";
    world_frame_ = "world";
    camera_frame_ = "camera";
    imu_frame_ = "imu";
    nh_private_.param("map_frame", map_frame_, map_frame_);
    nh_private_.param("world_frame", world_frame_, world_frame_);
    nh_private_.param("camera_frame", camera_frame_, camera_frame_);
    nh_private_.param("imu_frame", imu_frame_, imu_frame_);

    //CUSTOMIZATION
    nh_private_.param("imu_offset", imu_offset_, 0.0);
    nh_private_.param("cam0_offset", cam0_offset_, 0.0);
    nh_private_.param("cam1_offset", cam1_offset_, 0.0);
    nh_private_.param("cam2_offset", cam2_offset_, 0.0);
    nh_private_.param("histogram_equalize_8bit_images", histogram_equalize_8bit_images_, false);
    if (histogram_equalize_8bit_images_)
    {
      nh_private_.param("clahe_clip_limit", clahe_clip_limit_, 4.0);
      nh_private_.param("clahe_grid_size", clahe_grid_size_, 64.0);
      clahe = cv::createCLAHE();
      clahe->setClipLimit(clahe_clip_limit_);
      clahe->setTilesGridSize(cv::Size(clahe_grid_size_, clahe_grid_size_));
    }
    nh_private_.param("resize_input_image", resize_input_image_, false);
    nh_private_.param("resize_factor", resize_factor_, 0.5);
    if (resize_factor_ > 1.0)
      resize_factor_ = 1.0;
    //CUSTOMIZATION

    // Initialize messages
    transformMsg_.header.frame_id = world_frame_;
    transformMsg_.child_frame_id = imu_frame_;
    odometryMsg_.header.frame_id = world_frame_;
    odometryMsg_.child_frame_id = imu_frame_;
    msgSeq_ = 1;
    for (int camID = 0; camID < mtState::nCam_; camID++)
    {
      extrinsicsMsg_[camID].header.frame_id = imu_frame_;
    }
    imuBiasMsg_.header.frame_id = world_frame_;
    imuBiasMsg_.orientation.x = 0;
    imuBiasMsg_.orientation.y = 0;
    imuBiasMsg_.orientation.z = 0;
    imuBiasMsg_.orientation.w = 1;
    for (int i = 0; i < 9; i++)
    {
      imuBiasMsg_.orientation_covariance[i] = 0.0;
    }

    // PointCloud message.
    pclMsg_.header.frame_id = imu_frame_;
    pclMsg_.height = 1;             // Unordered point cloud.
    pclMsg_.width = mtState::nMax_; // Number of features/points.
    const int nFieldsPcl = 18;
    std::string namePcl[nFieldsPcl] = {"id", "camId", "rgb", "status", "x", "y", "z", "b_x", "b_y", "b_z", "d", "c_00", "c_01", "c_02", "c_11", "c_12", "c_22", "c_d"};
    int sizePcl[nFieldsPcl] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    int countPcl[nFieldsPcl] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int datatypePcl[nFieldsPcl] = {sensor_msgs::PointField::INT32, sensor_msgs::PointField::INT32, sensor_msgs::PointField::UINT32, sensor_msgs::PointField::UINT32,
                                   sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32,
                                   sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32,
                                   sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32};
    pclMsg_.fields.resize(nFieldsPcl);
    int byteCounter = 0;
    for (int i = 0; i < nFieldsPcl; i++)
    {
      pclMsg_.fields[i].name = namePcl[i];
      pclMsg_.fields[i].offset = byteCounter;
      pclMsg_.fields[i].count = countPcl[i];
      pclMsg_.fields[i].datatype = datatypePcl[i];
      byteCounter += sizePcl[i] * countPcl[i];
    }
    pclMsg_.point_step = byteCounter;
    pclMsg_.row_step = pclMsg_.point_step * pclMsg_.width;
    pclMsg_.data.resize(pclMsg_.row_step * pclMsg_.height);
    pclMsg_.is_dense = false;

    // PointCloud message.
    patchMsg_.header.frame_id = "camera0";
    patchMsg_.height = 1;             // Unordered point cloud.
    patchMsg_.width = mtState::nMax_; // Number of features/points.
    const int nFieldsPatch = 5;
    std::string namePatch[nFieldsPatch] = {"id", "patch", "dx", "dy", "error"};
    int sizePatch[nFieldsPatch] = {4, 4, 4, 4, 4};
    int countPatch[nFieldsPatch] = {1, mtState::nLevels_ * mtState::patchSize_ * mtState::patchSize_, mtState::nLevels_ * mtState::patchSize_ * mtState::patchSize_, mtState::nLevels_ * mtState::patchSize_ * mtState::patchSize_, mtState::nLevels_ * mtState::patchSize_ * mtState::patchSize_};
    int datatypePatch[nFieldsPatch] = {sensor_msgs::PointField::INT32, sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32, sensor_msgs::PointField::FLOAT32};
    patchMsg_.fields.resize(nFieldsPatch);
    byteCounter = 0;
    for (int i = 0; i < nFieldsPatch; i++)
    {
      patchMsg_.fields[i].name = namePatch[i];
      patchMsg_.fields[i].offset = byteCounter;
      patchMsg_.fields[i].count = countPatch[i];
      patchMsg_.fields[i].datatype = datatypePatch[i];
      byteCounter += sizePatch[i] * countPatch[i];
    }
    patchMsg_.point_step = byteCounter;
    patchMsg_.row_step = patchMsg_.point_step * patchMsg_.width;
    patchMsg_.data.resize(patchMsg_.row_step * patchMsg_.height);
    patchMsg_.is_dense = false;

    // Marker message (vizualization of uncertainty)
    markerMsg_.header.frame_id = imu_frame_;
    markerMsg_.id = 0;
    markerMsg_.type = visualization_msgs::Marker::LINE_LIST;
    markerMsg_.action = visualization_msgs::Marker::ADD;
    markerMsg_.pose.position.x = 0;
    markerMsg_.pose.position.y = 0;
    markerMsg_.pose.position.z = 0;
    markerMsg_.pose.orientation.x = 0.0;
    markerMsg_.pose.orientation.y = 0.0;
    markerMsg_.pose.orientation.z = 0.0;
    markerMsg_.pose.orientation.w = 1.0;
    markerMsg_.scale.x = 0.04; // Line width.
    markerMsg_.color.a = 1.0;
    markerMsg_.color.r = 0.0;
    markerMsg_.color.g = 1.0;
    markerMsg_.color.b = 0.0;

    // Bsp: visualization extras
    BSP_frustumMsg_.header.frame_id = "/world";
    BSP_frustumMsg_.ns = "frustum";
    BSP_frustumMsg_.id = 1;
    BSP_frustumMsg_.type = visualization_msgs::Marker::LINE_STRIP;
    BSP_frustumMsg_.action = visualization_msgs::Marker::ADD;
    BSP_frustumMsg_.scale.x = 0.015;
    BSP_frustumMsg_.color.a = 1.0;
    if (mpFilter_->bspFilter_)
    {
      BSP_frustumMsg_.color.r = 1.0;
      BSP_frustumMsg_.color.b = 1.0;
    }
    else
    {
      BSP_frustumMsg_.color.g = 1.0;
    }
    BSP_bearingArrowArrayMsg_.markers.resize(mtState::nMax_);
    for (int i = 0; i < mtState::nMax_; i++)
    {
      visualization_msgs::Marker &BSP_bearingArrow_i = BSP_bearingArrowArrayMsg_.markers[i];
      BSP_bearingArrow_i.ns = "bearing_arrow";
      BSP_bearingArrow_i.id = i;
      BSP_bearingArrow_i.type = visualization_msgs::Marker::ARROW;
      BSP_bearingArrow_i.action = visualization_msgs::Marker::ADD;
      BSP_bearingArrow_i.scale.y = 0.01;
      BSP_bearingArrow_i.scale.z = 0.01;
    }
  }

  /** \brief Destructor
   */
  virtual ~RovioNode() {}

  /** \brief Tests the functionality of the rovio node.
   *
   *  @todo debug with   doVECalibration = false and depthType = 0
   */
  void makeTest()
  {
    mtFilterState *mpTestFilterState = new mtFilterState();
    *mpTestFilterState = mpFilter_->init_;
    mpTestFilterState->setCamera(&mpFilter_->multiCamera_);
    mtState &testState = mpTestFilterState->state_;
    unsigned int s = 2;
    testState.setRandom(s);
    predictionMeas_.setRandom(s);
    imgUpdateMeas_.setRandom(s);

    LWF::NormalVectorElement tempNor;
    for (int i = 0; i < mtState::nMax_; i++)
    {
      testState.CfP(i).camID_ = 0;
      tempNor.setRandom(s);
      if (tempNor.getVec()(2) < 0)
      {
        tempNor.boxPlus(Eigen::Vector2d(3.14, 0), tempNor);
      }
      testState.CfP(i).set_nor(tempNor);
      testState.CfP(i).trackWarping_ = false;
      tempNor.setRandom(s);
      if (tempNor.getVec()(2) < 0)
      {
        tempNor.boxPlus(Eigen::Vector2d(3.14, 0), tempNor);
      }
      testState.aux().feaCoorMeas_[i].set_nor(tempNor, true);
      testState.aux().feaCoorMeas_[i].mpCamera_ = &mpFilter_->multiCamera_.cameras_[0];
      testState.aux().feaCoorMeas_[i].camID_ = 0;
    }
    testState.CfP(0).camID_ = mtState::nCam_ - 1;
    mpTestFilterState->fsm_.setAllCameraPointers();

    // Prediction
    std::cout << "Testing Prediction" << std::endl;
    mpFilter_->mPrediction_.testPredictionJacs(testState, predictionMeas_, 1e-8, 1e-6, 0.1);

    // Update
    if (!mpImgUpdate_->useDirectMethod_)
    {
      std::cout << "Testing Update (can sometimes exhibit large absolut errors due to the float precision)" << std::endl;
      for (int i = 0; i < (std::min((int)mtState::nMax_, 2)); i++)
      {
        testState.aux().activeFeature_ = i;
        testState.aux().activeCameraCounter_ = 0;
        mpImgUpdate_->testUpdateJacs(testState, imgUpdateMeas_, 1e-4, 1e-5);
        testState.aux().activeCameraCounter_ = mtState::nCam_ - 1;
        mpImgUpdate_->testUpdateJacs(testState, imgUpdateMeas_, 1e-4, 1e-5);
      }
    }

    // Testing CameraOutputCF and CameraOutputCF
    std::cout << "Testing cameraOutputCF" << std::endl;
    cameraOutputCT_.testTransformJac(testState, 1e-8, 1e-6);
    std::cout << "Testing imuOutputCF" << std::endl;
    imuOutputCT_.testTransformJac(testState, 1e-8, 1e-6);
    std::cout << "Testing attitudeToYprCF" << std::endl;
    rovio::AttitudeToYprCT attitudeToYprCF;
    attitudeToYprCF.testTransformJac(1e-8, 1e-6);

    // Testing TransformFeatureOutputCT
    std::cout << "Testing transformFeatureOutputCT" << std::endl;
    transformFeatureOutputCT_.setFeatureID(0);
    if (mtState::nCam_ > 1)
    {
      transformFeatureOutputCT_.setOutputCameraID(1);
      transformFeatureOutputCT_.testTransformJac(testState, 1e-8, 1e-5);
    }
    transformFeatureOutputCT_.setOutputCameraID(0);
    transformFeatureOutputCT_.testTransformJac(testState, 1e-8, 1e-5);

    // Testing LandmarkOutputImuCT
    std::cout << "Testing LandmarkOutputImuCT" << std::endl;
    landmarkOutputImuCT_.setFeatureID(0);
    landmarkOutputImuCT_.testTransformJac(testState, 1e-8, 1e-5);

    // Getting featureOutput for next tests
    transformFeatureOutputCT_.transformState(testState, featureOutput_);
    if (!featureOutput_.c().isInFront())
    {
      featureOutput_.c().set_nor(featureOutput_.c().get_nor().rotated(QPD(0.0, 1.0, 0.0, 0.0)), false);
    }

    // Testing FeatureOutputReadableCT
    std::cout << "Testing FeatureOutputReadableCT" << std::endl;
    featureOutputReadableCT_.testTransformJac(featureOutput_, 1e-8, 1e-5);

    // Testing pixelOutputCT
    rovio::PixelOutputCT pixelOutputCT;
    std::cout << "Testing pixelOutputCT (can sometimes exhibit large absolut errors due to the float precision)" << std::endl;
    pixelOutputCT.testTransformJac(featureOutput_, 1e-4, 1.0); // Reduces accuracy due to float and strong camera distortion

    // Testing ZeroVelocityUpdate_
    std::cout << "Testing zero velocity update" << std::endl;
    mpImgUpdate_->zeroVelocityUpdate_.testJacs();

    // Testing PoseUpdate
    if (!mpPoseUpdate_->noFeedbackToRovio_)
    {
      std::cout << "Testing pose update" << std::endl;
      mpPoseUpdate_->testUpdateJacs(1e-8, 1e-5);
    }

    delete mpTestFilterState;
  }

  /** \brief Bsp: ROS service handler to pack and send belief (filter) state.
  *
  *  @param request  - \ref bsp_msgs::BSP_SrvSendFilterState::Request
  *  @param response  - \ref bsp_msgs::BSP_SrvSendFilterState::Response
  */
  bool BSP_servFilterStateCallback(bsp_msgs::BSP_SrvSendFilterState::Request &request,
                                   bsp_msgs::BSP_SrvSendFilterState::Response &response)
  {
    const mtFilterState &filterState = mpFilter_->safe_;
    const mtState &state = mpFilter_->safe_.state_;
    state.updateMultiCameraExtrinsics(&mpFilter_->multiCamera_);
    const MXD &cov = mpFilter_->safe_.cov_;
    imuOutputCT_.transformState(state, imuOutput_);

    bsp_msgs::ROVIO_StateMsg state_msg;

    state_msg.nMax = mtState::nMax_;
    //state_msg.nLevels = mtState::nLevels_; //not part of BSP_StateMsg
    //state_msg.patchSize = mtState::patchSize_; //not part of BSP_StateMsg
    state_msg.nCam = mtState::nCam_;
    //state_msg.nPose = mtState::nPose_; //not part of BSP_StateMsg
    tf::vectorEigenToMsg(state.WrWM(), state_msg.pos_WrWM);
    tf::vectorEigenToMsg(state.MvM(), state_msg.vel_MvM);
    tf::vectorEigenToMsg(state.acb(), state_msg.acb);
    tf::vectorEigenToMsg(state.gyb(), state_msg.gyb);
    const QPD &state_qWM = state.qWM();
    tf::quaternionEigenToMsg(Eigen::Quaterniond(state_qWM.w(), state_qWM.x(), state_qWM.y(), state_qWM.z()), state_msg.att_qWM);
    state_msg.vep_MrMC.resize(mtState::nCam_);
    state_msg.vea_qCM.resize(mtState::nCam_);
    for (unsigned int i = 0; i < mtState::nCam_; ++i)
    {
      tf::vectorEigenToMsg(state.MrMC(i), state_msg.vep_MrMC.at(i));
      const QPD &state_qCM_i = state.qCM(i);
      tf::quaternionEigenToMsg(Eigen::Quaterniond(state_qCM_i.w(), state_qCM_i.x(), state_qCM_i.y(), state_qCM_i.z()), state_msg.vea_qCM.at(i));
    }

    std::vector<bsp_msgs::ROVIO_RobocentricFeatureElementMsg> robocentricFeatureElement_msgVec;
    robocentricFeatureElement_msgVec.resize(mtState::nMax_);
    for (unsigned int i = 0; i < mtState::nMax_; ++i)
    {
      if (filterState.fsm_.isValid_[i])
      {
        const FeatureDistance &state_dep = state.dep(i);
        const FeatureCoordinates &state_CfP = state.CfP(i);
        //robocentricFeatureElement_msgVec.at(i).distance_type_enum = state_dep.getType();
        robocentricFeatureElement_msgVec.at(i).p = state_dep.getDistance();
        const cv::Point2f &state_CfP_c = state_CfP.get_c();
        robocentricFeatureElement_msgVec.at(i).c.x = state_CfP_c.x;
        robocentricFeatureElement_msgVec.at(i).c.y = state_CfP_c.y;
        robocentricFeatureElement_msgVec.at(i).valid_c = state_CfP.valid_c_;
        robocentricFeatureElement_msgVec.at(i).valid_nor = state_CfP.valid_nor_;
        tf::quaternionEigenToMsg(Eigen::Quaterniond(state_CfP.nor_.q_.w(), state_CfP.nor_.q_.x(), state_CfP.nor_.q_.y(), state_CfP.nor_.q_.z()), robocentricFeatureElement_msgVec.at(i).q);
        robocentricFeatureElement_msgVec.at(i).camID = filterState.fsm_.features_[i].mpCoordinates_->camID_;
        robocentricFeatureElement_msgVec.at(i).warp_c_2d = {state_CfP.warp_c_(0, 0), state_CfP.warp_c_(0, 1), state_CfP.warp_c_(1, 0), state_CfP.warp_c_(1, 1)};
        robocentricFeatureElement_msgVec.at(i).valid_warp_c = state_CfP.valid_warp_c_;
        robocentricFeatureElement_msgVec.at(i).warp_nor_2d = {state_CfP.warp_nor_(0, 0), state_CfP.warp_nor_(0, 1), state_CfP.warp_nor_(1, 0), state_CfP.warp_nor_(1, 1)};
        robocentricFeatureElement_msgVec.at(i).valid_warp_nor = state_CfP.valid_warp_nor_;
        robocentricFeatureElement_msgVec.at(i).isWarpIdentity = state_CfP.isWarpIdentity_;
        robocentricFeatureElement_msgVec.at(i).trackWarping = state_CfP.trackWarping_;
      }
    }
    state_msg.fea = robocentricFeatureElement_msgVec;

    if (mpPoseUpdate_->inertialPoseIndex_ >= 0)
    {
      const V3D IrIW = state.poseLin(mpPoseUpdate_->inertialPoseIndex_);
      const QPD qWI = state.poseRot(mpPoseUpdate_->inertialPoseIndex_);
      tf::vectorEigenToMsg(IrIW, state_msg.pop_IrIW);
      tf::quaternionEigenToMsg(Eigen::Quaterniond(qWI.w(), qWI.x(), qWI.y(), qWI.z()), state_msg.poa_qWI);
    }
    //aux

    bsp_msgs::ROVIO_FilterStateMsg filterState_msg;
    filterState_msg.header.seq = 0;
    filterState_msg.header.frame_id = imu_frame_;
    filterState_msg.header.stamp = ros::Time(filterState.t_);
    filterState_msg.t = filterState.t_;
    //if (filterState.mode_==LWF::FilteringMode::ModeEKF) filterState_msg.mode=std::string("ModeEKF"); else if (filterState.mode_==LWF::FilteringMode::ModeUKF) filterState_msg.mode=std::string("ModeUKF"); else if (filterState.mode_==LWF::FilteringMode::ModeIEKF) filterState_msg.mode=std::string("ModeIEKF"); else filterState_msg.mode=std::string("Unknown"); //not part of BSP_FilterStateMsg
    //filterState_msg.usePredictionMerge = filterState.usePredictionMerge_; //not part of BSP_FilterStateMsg
    filterState_msg.state = state_msg;
    tf::matrixEigenToMsg(filterState.cov_, filterState_msg.cov);
    filterState_msg.fsm.maxIdx = filterState.fsm_.maxIdx_;
    filterState_msg.fsm.isValid.resize(mtState::nMax_);
    for (unsigned int i = 0; i < mtState::nMax_; ++i)
      filterState_msg.fsm.isValid.at(i) = filterState.fsm_.isValid_[i];

    MXD covSubMat;
    for (unsigned int i = 0; i < mtState::nMax_; ++i)
      mpFilter_->bsp_featureParams_[i].getParams(i, filterState, mpFilter_->octree_);
    mpFilter_->calcCovSubMat(filterState.cov_, covSubMat);
    filterState_msg.opt_metric = mpFilter_->calcDopt(covSubMat);
    filterState_msg.landmarks_pcl.clear();
    for (unsigned int i = 0; i < mtState::nMax_; ++i)
    {
      if (filterState.fsm_.isValid_[i])
      {
        const V3D &e_landmark_pt = mpFilter_->bsp_featureParams_[i].bsp_w3D();
        geometry_msgs::Point32 landmark_pt;
        landmark_pt.x = e_landmark_pt.x();
        landmark_pt.y = e_landmark_pt.y();
        landmark_pt.z = e_landmark_pt.z();
        filterState_msg.landmarks_pcl.push_back(landmark_pt);
      }
    }

    response.filterState = filterState_msg;
    return true;
  }

  /** \brief Bsp: ROS service handler to get and propagate, then pack and send belief (filter) state.
  *
  *  @param request  - \ref bsp_msgs::BSP_SrvPropagateFilterState::Request
  *  @param response  - \ref bsp_msgs::BSP_SrvPropagateFilterState::Response
  */
  bool BSP_servPropagateFilterStateCallback(bsp_msgs::BSP_SrvPropagateFilterState::Request &request,
                                            bsp_msgs::BSP_SrvPropagateFilterState::Response &response)
  {
    if (request.vecTrajectoryReferenceMsg.empty() || request.filterStateMsgInit.state.nMax != mtState::nMax_ || request.filterStateMsgInit.state.nCam != mtState::nCam_)
      return false;

    // 1: get data
    std::string bsp_frame = request.vecTrajectoryReferenceMsg.at(0).header.frame_id;
    /*
     * Bsp: Have to get last available tf, because of tree-based ordering of planning steps (step2 will start at the end of step1, potentially in the future, where tf is unavailable).
     */
    try
    {
      tf::StampedTransform bsp_tf;
      bsp_tl_.lookupTransform(bsp_frame, imu_frame_, ros::Time(0), bsp_tf);
      bsp_T_ = bsp_tf.getOrigin();
      bsp_Q_ = bsp_tf.getRotation();
    }
    catch (const tf::TransformException &ex)
    {
      ROS_ERROR_STREAM("Error getting tf data: " << ex.what());
      return false;
    }

    mpFilter_->init_.t_ = request.filterStateMsgInit.t;
    try
    {
      bsp::msgToMatrixEigen(request.filterStateMsgInit.cov, mpFilter_->init_.cov_);
    }
    catch (const std::runtime_error &ex)
    {
      ROS_ERROR_STREAM("Error mapping std_msgs data to eigen: " << ex.what());
      return false;
    }

    tf::vectorMsgToEigen(request.filterStateMsgInit.state.pos_WrWM, mpFilter_->init_.state_.WrWM());
    tf::vectorMsgToEigen(request.filterStateMsgInit.state.vel_MvM, mpFilter_->init_.state_.MvM());
    tf::vectorMsgToEigen(request.filterStateMsgInit.state.acb, mpFilter_->init_.state_.acb());
    tf::vectorMsgToEigen(request.filterStateMsgInit.state.gyb, mpFilter_->init_.state_.gyb());
    Eigen::Quaterniond e_qWM;
    tf::quaternionMsgToEigen(request.filterStateMsgInit.state.att_qWM, e_qWM);
    mpFilter_->init_.state_.qWM().setValues(e_qWM.w(), e_qWM.x(), e_qWM.y(), e_qWM.z());
    for (int i = 0; i < mtState::nCam_; ++i)
    {
      tf::vectorMsgToEigen(request.filterStateMsgInit.state.vep_MrMC.at(i), mpFilter_->init_.state_.MrMC(i));
      Eigen::Quaterniond e_qCM_i;
      tf::quaternionMsgToEigen(request.filterStateMsgInit.state.vea_qCM.at(i), e_qCM_i);
      mpFilter_->init_.state_.qCM(i).setValues(e_qCM_i.w(), e_qCM_i.x(), e_qCM_i.y(), e_qCM_i.z());
    }

    if (mpPoseUpdate_->inertialPoseIndex_ >= 0)
    {
      tf::vectorMsgToEigen(request.filterStateMsgInit.state.pop_IrIW, mpFilter_->init_.state_.poseLin(mpPoseUpdate_->inertialPoseIndex_));
      Eigen::Quaterniond e_qWI;
      tf::quaternionMsgToEigen(request.filterStateMsgInit.state.poa_qWI, e_qWI);
      mpFilter_->init_.state_.poseRot(mpPoseUpdate_->inertialPoseIndex_).setValues(e_qWI.w(), e_qWI.x(), e_qWI.y(), e_qWI.z());
    }
    init_state_.WrWM_ = mpFilter_->init_.state_.WrWM();
    init_state_.qMW_ = mpFilter_->init_.state_.qWM().inverted();

    mpFilter_->init_.fsm_.maxIdx_ = request.filterStateMsgInit.fsm.maxIdx;
    for (unsigned int i = 0; i < mtState::nMax_; ++i)
    {
      mpFilter_->init_.fsm_.isValid_[i] = request.filterStateMsgInit.fsm.isValid.at(i);
    }
    for (unsigned int i = 0; i < mtState::nMax_; ++i)
    {
      if (request.filterStateMsgInit.fsm.isValid.at(i))
      {
        const bsp_msgs::ROVIO_RobocentricFeatureElementMsg &fea_i = request.filterStateMsgInit.state.fea.at(i);
        FeatureDistance &init_state_dep = mpFilter_->init_.state_.dep(i);
        //init_state_dep.setType(fea_i.distance_type_enum);
        init_state_dep.setParameter(fea_i.p);
        FeatureCoordinates &init_state_CfP = mpFilter_->init_.state_.CfP(i);
        init_state_CfP.set_c(cv::Point2f(fea_i.c.x, fea_i.c.y), false);
        init_state_CfP.set_nor(LWF::NormalVectorElement(QPD(fea_i.q.w, fea_i.q.x, fea_i.q.y, fea_i.q.z)), false);
        init_state_CfP.valid_c_ = fea_i.valid_c;
        init_state_CfP.valid_nor_ = fea_i.valid_nor;
        mpFilter_->init_.fsm_.features_[i].mpCoordinates_->camID_ = request.filterStateMsgInit.state.fea.at(i).camID;
        init_state_CfP.warp_c_ << fea_i.warp_c_2d.at(0), fea_i.warp_c_2d.at(1), fea_i.warp_c_2d.at(2), fea_i.warp_c_2d.at(3);
        init_state_CfP.valid_warp_c_ = request.filterStateMsgInit.state.fea.at(i).valid_warp_c;
        init_state_CfP.warp_nor_ << fea_i.warp_nor_2d.at(0), fea_i.warp_nor_2d.at(1), fea_i.warp_nor_2d.at(2), fea_i.warp_nor_2d.at(3);
        init_state_CfP.valid_warp_nor_ = request.filterStateMsgInit.state.fea.at(i).valid_warp_nor;
        init_state_CfP.isWarpIdentity_ = request.filterStateMsgInit.state.fea.at(i).isWarpIdentity;
        init_state_CfP.trackWarping_ = request.filterStateMsgInit.state.fea.at(i).trackWarping;
      }
    }
    //request.filterStateMsgInit.opt_metric;

    if (!request.filterStateMap.data.empty())
    {
      /*
       * Bsp: No need to re-acquire on every re-planning call, work with last available map 
       */
      if (bsp_rootmap_stamp_ < request.filterStateMap.header.stamp)
      {
        ROS_INFO("New filterStateMap data detected in bsp propagation request!");
        delete mpFilter_->octree_;
        if (request.filterStateMap.binary)
        {
          // Bsp: The more efficient implementation will carry over only the binary map, depends on bsp_planner, depends on suggested volumetric planning modifications
          mpFilter_->octree_ = dynamic_cast<octomap::OcTree *>(octomap_msgs::binaryMsgToMap(request.filterStateMap));
        }
        else
        {
          // Bsp: Handle the less efficient implementation
          mpFilter_->octree_ = dynamic_cast<octomap::OcTree *>(octomap_msgs::fullMsgToMap(request.filterStateMap));
        }
        if (mpFilter_->octree_ != nullptr)
        {
          bsp_rootmap_stamp_ = request.filterStateMap.header.stamp;
          mpFilter_->octree_->prune();
        }
        else
        {
          ROS_WARN("octomap_msgs::binaryMsgToMap() did not manage to derive a map...");
        }
      }
      else
      {
        ROS_WARN("Old filterStateMap data detected in bsp propagation request...");
      }
    }

    // 2: propagate data
    //requestReset();
    requestResetToPose(init_state_.WrWM_, init_state_.qMW_);
    //filterState.fsm_.allocateMissing();
    mpFilter_->init_.fsm_.setAllCameraPointers(); //TODO: why after 1st bsp iteration state.CfP(i).mpCamera_!=nullptr for all features?

    const double g = -mpFilter_->mPrediction_.g_[2];

    sensor_msgs::Imu::Ptr predictionImu_msg = boost::make_shared<sensor_msgs::Imu>();
    const unsigned int bsp_propSimLimit = 10 / mpFilter_->bsp_Ts_;
    for (unsigned int i = 1;; ++i)
    {
      if (i >= bsp_propSimLimit)
      {
        ROS_ERROR("Waypoint NOT reached...");
        break;
      }
      const bsp_msgs::BSP_TrajectoryReferenceMsg &trajectoryReference_msg = request.vecTrajectoryReferenceMsg.at(0);

      const mtFilterState &filterState = mpFilter_->safe_;
      const mtState &state = mpFilter_->safe_.state_;

      const V3D &pos_WrWM = state.WrWM();
      const V3D &vel_MvM = state.MvM();
      const V3D &acb = state.acb();
      const V3D &gyb = state.gyb();
      const QPD &att_qMW = state.qWM().inverted();
      const tf::Vector3 tf_pos_WrWM(pos_WrWM.x(), pos_WrWM.y(), pos_WrWM.z());
      const tf::Quaternion tf_att_qMW(att_qMW.x(), att_qMW.y(), att_qMW.z(), att_qMW.w());
      tfScalar tf_att_qMW_roll, tf_att_qMW_pitch, tf_att_qMW_yaw;
      tf::Matrix3x3(tf_att_qMW).getRPY(tf_att_qMW_roll, tf_att_qMW_pitch, tf_att_qMW_yaw);

      const tf::Vector3 tf_pos_WrWM_ref(trajectoryReference_msg.pose.position.x, trajectoryReference_msg.pose.position.y, trajectoryReference_msg.pose.position.z);
      const tf::Quaternion tf_att_qMW_ref(trajectoryReference_msg.pose.orientation.x, trajectoryReference_msg.pose.orientation.y, trajectoryReference_msg.pose.orientation.z, trajectoryReference_msg.pose.orientation.w);
      tfScalar tf_att_qMW_roll_ref, tf_att_qMW_pitch_ref, tf_att_qMW_yaw_ref;
      tf::Matrix3x3(tf_att_qMW_ref).getRPY(tf_att_qMW_roll_ref, tf_att_qMW_pitch_ref, tf_att_qMW_yaw_ref);

      const tf::Vector3 tf_pos_WrWM_err = tf_pos_WrWM_ref - tf_pos_WrWM;
      const tfScalar tf_att_qMW_yaw_err_unwrapped = tf_att_qMW_yaw_ref - tf_att_qMW_yaw;
      const tfScalar tf_att_qMW_yaw_err = (tf_att_qMW_yaw_err_unwrapped > M_PI) ? (-2.0 * M_PI) : (tf_att_qMW_yaw_err_unwrapped < -M_PI ? 2.0 * M_PI : 0) + tf_att_qMW_yaw_err_unwrapped;

      if (tf_pos_WrWM_err.length2() <= 0.05 * 0.05 && vel_MvM.squaredNorm() <= 0.1 * 0.1 && std::abs(tf_att_qMW_yaw_err) <= 0.017453293)
      {
        ROS_INFO("Waypoint reached!");
        break;
      }

      const tf::Vector3 tf_pos_WrWM_err_BFF = tf::Transform(tf_att_qMW, tf::Vector3(0.0, 0.0, 0.0)).inverse() * tf_pos_WrWM_err;
      tf_att_qMW_roll_ref = std::min(std::max(-mpFilter_->rollpitch_max_, -(mpFilter_->xy_P_ * tf_pos_WrWM_err_BFF.getY() + mpFilter_->xy_D_ * vel_MvM.y())), mpFilter_->rollpitch_max_);
      tf_att_qMW_pitch_ref = std::min(std::max(-mpFilter_->rollpitch_max_, (mpFilter_->xy_P_ * tf_pos_WrWM_err_BFF.getX() + mpFilter_->xy_D_ * vel_MvM.x())), mpFilter_->rollpitch_max_);
      const tfScalar tf_att_qMW_roll_err = tf_att_qMW_roll_ref - tf_att_qMW_roll;
      const tfScalar tf_att_qMW_pitch_err = tf_att_qMW_pitch_ref - tf_att_qMW_pitch;

      predictionImu_msg->header.frame_id = "body";
      predictionImu_msg->header.stamp = ros::Time(mpFilter_->init_.t_ + mpFilter_->bsp_Ts_ * i);
      predictionImu_msg->linear_acceleration.x = acb.x() + sin(tf_att_qMW_pitch) * g + mpFilter_->xyz_damp_ * vel_MvM.x();
      predictionImu_msg->linear_acceleration.y = acb.y() + -sin(tf_att_qMW_roll) * g + mpFilter_->xyz_damp_ * vel_MvM.y();
      predictionImu_msg->linear_acceleration.z = acb.z() + g + (mpFilter_->z_P_ * tf_pos_WrWM_err_BFF.getZ() + mpFilter_->z_D_ * vel_MvM.z()) + mpFilter_->xyz_damp_ * vel_MvM.z();
      predictionImu_msg->angular_velocity.x = gyb.x() + std::min(std::max(-mpFilter_->xy_gyr_max_, mpFilter_->xy_gyr_P_ * tf_att_qMW_roll_err), mpFilter_->xy_gyr_max_);
      predictionImu_msg->angular_velocity.y = gyb.y() + std::min(std::max(-mpFilter_->xy_gyr_max_, mpFilter_->xy_gyr_P_ * tf_att_qMW_pitch_err), mpFilter_->xy_gyr_max_);
      predictionImu_msg->angular_velocity.z = gyb.z() + std::min(std::max(-mpFilter_->z_gyr_max_, mpFilter_->z_gyr_P_ * tf_att_qMW_yaw_err), mpFilter_->z_gyr_max_);

      imuCallback(predictionImu_msg);
      BSP_landmarkCallback();
    }

    // 2: pack and send data
    mtFilterState &filterState = mpFilter_->safe_;
    mtState &state = mpFilter_->safe_.state_;
    state.updateMultiCameraExtrinsics(&mpFilter_->multiCamera_);
    MXD &cov = mpFilter_->safe_.cov_;
    imuOutputCT_.transformState(state, imuOutput_);

    ros::Time filterState_stamp = ros::Time(filterState.t_);

    bsp_msgs::ROVIO_StateMsg state_msg;

    state_msg.nMax = state.nMax_;
    //state_msg.nLevels = state.nLevels_;
    //state_msg.patchSize = state.patchSize_;
    state_msg.nCam = state.nCam_;
    //state_msg.nPose = state.nPose_;
    tf::vectorEigenToMsg(state.WrWM(), state_msg.pos_WrWM);
    tf::vectorEigenToMsg(state.MvM(), state_msg.vel_MvM);
    tf::vectorEigenToMsg(state.acb(), state_msg.acb);
    tf::vectorEigenToMsg(state.gyb(), state_msg.gyb);
    tf::quaternionEigenToMsg(Eigen::Quaterniond(state.qWM().w(), state.qWM().x(), state.qWM().y(), state.qWM().z()), state_msg.att_qWM);
    state_msg.vep_MrMC.resize(mtState::nCam_);
    state_msg.vea_qCM.resize(mtState::nCam_);
    for (int i = 0; i < mtState::nCam_; ++i)
    {
      tf::vectorEigenToMsg(state.MrMC(i), state_msg.vep_MrMC.at(i));
      const QPD &state_qCM_i = state.qCM(i);
      tf::quaternionEigenToMsg(Eigen::Quaterniond(state_qCM_i.w(), state_qCM_i.x(), state_qCM_i.y(), state_qCM_i.z()), state_msg.vea_qCM.at(i));
    }

    std::vector<bsp_msgs::ROVIO_RobocentricFeatureElementMsg> robocentricFeatureElement_msgVec;
    robocentricFeatureElement_msgVec.resize(mtState::nMax_);
    for (unsigned int i = 0; i < mtState::nMax_; ++i)
    {
      if (request.filterStateMsgInit.fsm.isValid.at(i))
      {
        const FeatureDistance &state_dep = state.dep(i);
        /*const*/ FeatureCoordinates &state_CfP = state.CfP(i);
        double lmrk_d = state_dep.getDistance();
        Eigen::Quaterniond lmrk_q(state_CfP.nor_.q_.w(), state_CfP.nor_.q_.x(), state_CfP.nor_.q_.y(), state_CfP.nor_.q_.z());
        robocentricFeatureElement_msgVec.at(i).p = lmrk_d;
        tf::quaternionEigenToMsg(lmrk_q, robocentricFeatureElement_msgVec.at(i).q);
        const LWF::NormalVectorElement &lmrk_nor = LWF::NormalVectorElement(QPD(lmrk_q.w(), lmrk_q.x(), lmrk_q.y(), lmrk_q.z()));
        state_CfP.set_nor(lmrk_nor, false); //also sets valid_nor_ and unsets valid_c_
        //TODO: Could avoid 2x-checking if bearing depth is behind the camera, check also happens at get_c() call time, leave for now
        if (lmrk_nor.getVec()(2) <= 0)
        {
          robocentricFeatureElement_msgVec.at(i).c.x = -1;
          robocentricFeatureElement_msgVec.at(i).c.y = -1;
        }
        else
        {
          const cv::Point2f &state_CfP_c = state_CfP.get_c(); //also calls com_c_ which checks valid_nor_ and sets valid_c_
          robocentricFeatureElement_msgVec.at(i).c.x = state_CfP_c.x;
          robocentricFeatureElement_msgVec.at(i).c.y = state_CfP_c.y;
        }
        robocentricFeatureElement_msgVec.at(i).valid_nor = state_CfP.valid_nor_;
        robocentricFeatureElement_msgVec.at(i).valid_c = state_CfP.valid_c_;
        robocentricFeatureElement_msgVec.at(i).camID = filterState.fsm_.features_[i].mpCoordinates_->camID_;
        robocentricFeatureElement_msgVec.at(i).warp_c_2d = {state_CfP.warp_c_(0, 0), state_CfP.warp_c_(0, 1), state_CfP.warp_c_(1, 0), state_CfP.warp_c_(1, 1)};
        robocentricFeatureElement_msgVec.at(i).valid_warp_c = state_CfP.valid_warp_c_;
        robocentricFeatureElement_msgVec.at(i).warp_nor_2d = {state_CfP.warp_nor_(0, 0), state_CfP.warp_nor_(0, 1), state_CfP.warp_nor_(1, 0), state_CfP.warp_nor_(1, 1)};
        robocentricFeatureElement_msgVec.at(i).valid_warp_nor = state_CfP.valid_warp_nor_;
        robocentricFeatureElement_msgVec.at(i).isWarpIdentity = state_CfP.isWarpIdentity_;
        robocentricFeatureElement_msgVec.at(i).trackWarping = state_CfP.trackWarping_;
      }
    }
    state_msg.fea = robocentricFeatureElement_msgVec;

    if (mpPoseUpdate_->inertialPoseIndex_ >= 0)
    {
      Eigen::Vector3d IrIW = state.poseLin(mpPoseUpdate_->inertialPoseIndex_);
      QPD qWI = state.poseRot(mpPoseUpdate_->inertialPoseIndex_);
      tf::vectorEigenToMsg(IrIW, state_msg.pop_IrIW);
      tf::quaternionEigenToMsg(Eigen::Quaterniond(qWI.w(), qWI.x(), qWI.y(), qWI.z()), state_msg.poa_qWI);
    }
    //aux

    bsp_msgs::ROVIO_FilterStateMsg filterState_msg;
    filterState_msg.header.seq = bsp_planning_seq_;
    filterState_msg.header.frame_id = imu_frame_;
    filterState_msg.header.stamp = filterState_stamp;
    filterState_msg.t = filterState.t_;
    //if (filterState.mode_==LWF::FilteringMode::ModeEKF) filterState_msg.mode=std::string("ModeEKF"); else if (filterState.mode_==LWF::FilteringMode::ModeUKF) filterState_msg.mode=std::string("ModeUKF"); else if (filterState.mode_==LWF::FilteringMode::ModeIEKF) filterState_msg.mode=std::string("ModeIEKF"); else filterState_msg.mode=std::string("Unknown");
    //filterState_msg.usePredictionMerge = filterState.usePredictionMerge_;
    filterState_msg.state = state_msg;
    tf::matrixEigenToMsg(filterState.cov_, filterState_msg.cov);
    filterState_msg.fsm.maxIdx = filterState.fsm_.maxIdx_;
    filterState_msg.fsm.isValid.resize(mtState::nMax_);
    for (unsigned int i = 0; i < mtState::nMax_; ++i)
      filterState_msg.fsm.isValid.at(i) = filterState.fsm_.isValid_[i];

    MXD covSubMat;
    mpFilter_->calcCovSubMat(filterState.cov_, covSubMat);
    filterState_msg.opt_metric = mpFilter_->calcDopt(covSubMat);
    filterState_msg.landmarks_pcl = request.filterStateMsgInit.landmarks_pcl; //TODO: not pass entire landmarks_pcl from planning state to planning state

    response.filterStateMsgFinal = filterState_msg;

    response.filterStateMap_stamp = bsp_rootmap_stamp_;

    return true;
  }

  /** \brief Bsp: Callback for Landmark-Messages.
   */
  void BSP_landmarkCallback()
  {
    std::lock_guard<std::mutex> lock(m_filter_);
    if (init_state_.isInitialized())
    {
      mtFilterState &filterState = mpFilter_->safe_;
      mtState &state = mpFilter_->safe_.state_;

      ros::Time filterState_stamp = ros::Time::now();

      // Bsp: data respective to belief (filter) state
      for (unsigned int i = 0; i < mtState::nMax_; ++i)
        mpFilter_->bsp_featureParams_[i].getParams(i, filterState, mpFilter_->octree_);

      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, mtState::D_);
      rovio::FeatureOutput featureOutput;
      cv::Point2f c_temp;
      Eigen::Matrix2d c_J;

      Eigen::MatrixXd Hn = Eigen::Matrix2d::Identity();
      const double updateNoisePix = 2.0;
      Eigen::Matrix2d updnoiP = updateNoisePix * Eigen::Matrix2d::Identity();

      for (unsigned int i = 0; i < mtState::nMax_; ++i)
      {
        if (filterState.fsm_.isValid_[i] &&
            std::get<mt_BspFeatureParams::_bsp_fov>(mpFilter_->bsp_featureParams_[i]) &&
            std::get<mt_BspFeatureParams::_bsp_los>(mpFilter_->bsp_featureParams_[i]) == bsp::LoS::Free)
        {
          const unsigned int &camID = state.CfP(i).camID_;
          int activeCamCounter = state.aux().activeCameraCounter_;
          const unsigned int activeCamID = (activeCamCounter + camID) % mtState::nCam_;
          transformFeatureOutputCT_.setFeatureID(i);
          transformFeatureOutputCT_.setOutputCameraID(activeCamID);
          transformFeatureOutputCT_.transformState(state, featureOutput);

          filterState.fsm_.mpMultiCamera_->cameras_[activeCamID].bearingToPixel(featureOutput.c().get_nor(), c_temp, c_J);

          MXD::Index fea_idx = mtFilter::fea_0_idx_ + mtFilter::triplet_ * i;
          H = Eigen::MatrixXd::Zero(2, mtState::D_);
          H.block(0, fea_idx, 2, 2) = -c_J;

          Eigen::MatrixXd Pyinv;
          Eigen::MatrixXd K;

          Eigen::MatrixXd Py = H * filterState.cov_ * H.transpose() + Hn * updnoiP * Hn.transpose();
          Pyinv = Py.inverse();
          K = filterState.cov_ * H.transpose() * Pyinv;
          filterState.cov_ = filterState.cov_ - K * Py * K.transpose();
          filterState.cov_ = 0.5 * (filterState.cov_ + filterState.cov_.transpose()).eval();
        }
      }
    }
  }

  /** \brief Callback for IMU-Messages. Adds IMU measurements (as prediction measurements) to the filter.
   */
  void imuCallback(const sensor_msgs::Imu::ConstPtr &imu_msg)
  {
    std::lock_guard<std::mutex> lock(m_filter_);
    predictionMeas_.template get<mtPredictionMeas::_acc>() = Eigen::Vector3d(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    predictionMeas_.template get<mtPredictionMeas::_gyr>() = Eigen::Vector3d(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    if (init_state_.isInitialized())
    {
      mpFilter_->addPredictionMeas(predictionMeas_, imu_msg->header.stamp.toSec() + imu_offset_);
      updateAndPublish();
    }
    else
    {
      switch (init_state_.state_)
      {
      case FilterInitializationState::State::WaitForInitExternalPose:
      {
        std::cout << "-- Filter: Initializing using external pose ..." << std::endl;
        mpFilter_->resetWithPose(init_state_.WrWM_, init_state_.qMW_, imu_msg->header.stamp.toSec() + imu_offset_);
        break;
      }
      case FilterInitializationState::State::WaitForInitUsingAccel:
      {
        std::cout << "-- Filter: Initializing using accel. measurement ..." << std::endl;
        mpFilter_->resetWithAccelerometer(predictionMeas_.template get<mtPredictionMeas::_acc>(), imu_msg->header.stamp.toSec() + imu_offset_);
        break;
      }
      default:
      {
        std::cout << "Unhandeld initialization type." << std::endl;
        abort();
        break;
      }
      }

      std::cout << std::setprecision(12);
      std::cout << "-- Filter: Initialized at t = " << imu_msg->header.stamp.toSec() + imu_offset_ << std::endl;
      init_state_.state_ = FilterInitializationState::State::Initialized;
    }
  }

  /** \brief Image callback for the camera with ID 0
   *
   * @param img - Image message.
   * @todo generalize
   */
  void imgCallback0(const sensor_msgs::ImageConstPtr &img)
  {
    std::lock_guard<std::mutex> lock(m_filter_);
    imgCallback(img, 0);
  }

  /** \brief Image callback for the camera with ID 1
   *
   * @param img - Image message.
   * @todo generalize
   */
  void imgCallback1(const sensor_msgs::ImageConstPtr &img)
  {
    std::lock_guard<std::mutex> lock(m_filter_);
    if (mtState::nCam_ > 1)
      imgCallback(img, 1);
  }

  /** \brief Image callback for the camera with ID 2
   *
   * @param img - Image message.
   * @todo generalize
   */
  void imgCallback2(const sensor_msgs::ImageConstPtr &img)
  {
    std::lock_guard<std::mutex> lock(m_filter_);
    if (mtState::nCam_ > 2)
      imgCallback(img, 2);
  }

  /** \brief Image callback. Adds images (as update measurements) to the filter.
   *
   *   @param img   - Image message.
   *   @param camID - Camera ID.
   */
  void imgCallback(const sensor_msgs::ImageConstPtr &img, const int camID = 0)
  {
    // Get image from msg
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      //CUTOMIZATION
      //cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::TYPE_8UC1);
      cv_ptr = cv_bridge::toCvCopy(img, img->encoding); //smk: input image encoding doesn't matter as it gets converted to float point later
      //CUTOMIZATION
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    //CUSTOMIZATION - smk: offsets for all camera frames
    double cam_offset = 0.0;
    if (camID == 0)
      cam_offset = cam0_offset_;
    else if (camID == 1)
      cam_offset = cam1_offset_;
    else if (camID == 2)
      cam_offset = cam2_offset_;
    //CUSTOMIZATION

    //CUSTOMIZATION
    //Covert Color/Gray/16bit images to float
    cv::Mat cv_img;
    if (cv_ptr->encoding == "bgr8")
    {
      cv::cvtColor(cv_ptr->image, cv_img, CV_BGR2GRAY);
      cv_img.convertTo(cv_img, CV_32FC1);
    }
    else if (cv_ptr->encoding == "rgb8")
    {
      cv::cvtColor(cv_ptr->image, cv_img, CV_RGB2GRAY);
      cv_img.convertTo(cv_img, CV_32FC1);
    }
    else
      cv_ptr->image.convertTo(cv_img, CV_32FC1); //smk: convert incoming image to floating point value, this can deal with single channel 8 and 16 bit images

    //Image Resize
    if (resize_input_image_)
      cv::resize(cv_img, cv_img, cv::Size(), resize_factor_, resize_factor_); //INTER_LINEAR - a bilinear interpolation (used by default)

    //use CLAHE to histogram Equalize 8-bit intensity Images
    if (histogram_equalize_8bit_images_)
    {
      //Check if input image is actually 8-bit
      double imgMin, imgMax;
      cv::minMaxLoc(cv_img, &imgMin, &imgMax);
      if (imgMax <= 255.0)
      {
        cv::Mat inImg, outImg;
        cv_img.convertTo(inImg, CV_8UC1);
        clahe->apply(inImg, outImg);
        outImg.convertTo(cv_img, CV_32FC1);
      }
      else
        ROS_WARN_THROTTLE(5, "Histogram Equaliztion for 8-bit intensity images is turned on but input Image is not 8-bit");
    }
    //CUSTOMIZATION

    // MAPLAB
    if (pubDecimated_.getNumSubscribers() > 0)
    {
      if (resize_input_image_ || histogram_equalize_8bit_images_)
      {
        cv::Mat temp_image;
        cv_img.convertTo(temp_image, CV_8UC1);

        cv_bridge::CvImage img_bridge;

        img_bridge = cv_bridge::CvImage(img->header, sensor_msgs::image_encodings::MONO8, temp_image);
        img_bridge.toImageMsg(decimatedImage_); // from cv_bridge to sensor_msgs::Image

        pubDecimated_.publish(decimatedImage_); // ros::Publisher pub_img = node.advertise<sensor_msgs::Image>("topic", queuesize);
      }
    }
    //

    if (init_state_.isInitialized() && !cv_img.empty())
    {
      double msgTime = img->header.stamp.toSec() + cam_offset; //Adding offset for appropriate camera

      //CUSTOMIZATION - smk:removing check for multi-camera synchronization
      /*
      double msgTime = img->header.stamp.toSec();
      if (msgTime != imgUpdateMeas_.template get<mtImgMeas::_aux>().imgTime_)
      {
        for (int i = 0; i < mtState::nCam_; i++)
        {
          if (imgUpdateMeas_.template get<mtImgMeas::_aux>().isValidPyr_[i])
          {
            std::cout << "    \033[31mFailed Synchronization of Camera Frames, t = " << msgTime << "\033[0m" << std::endl;
          }
        }
        imgUpdateMeas_.template get<mtImgMeas::_aux>().reset(msgTime);
      }*/
      //CUSTOMIZATION

      imgUpdateMeas_.template get<mtImgMeas::_aux>().pyr_[camID].computeFromImage(cv_img, true); //smk: true = use openCV version of building pyramid
      imgUpdateMeas_.template get<mtImgMeas::_aux>().isValidPyr_[camID] = true;

      if (imgUpdateMeas_.template get<mtImgMeas::_aux>().areAllValid())
      {
        mpFilter_->template addUpdateMeas<0>(imgUpdateMeas_, msgTime);
        imgUpdateMeas_.template get<mtImgMeas::_aux>().reset(msgTime);
        updateAndPublish();
      }
    }
  }

  /** \brief Groundtruth callback for external groundtruth
   *
   *  @param transform - Groundtruth message.
   */
  void groundtruthCallback(const geometry_msgs::TransformStamped::ConstPtr &transform)
  {
    std::lock_guard<std::mutex> lock(m_filter_);
    if (init_state_.isInitialized())
    {
      poseUpdateMeas_.pos() = Eigen::Vector3d(transform->transform.translation.x, transform->transform.translation.y, transform->transform.translation.z);
      poseUpdateMeas_.att() = QPD(transform->transform.rotation.w, transform->transform.rotation.x, transform->transform.rotation.y, transform->transform.rotation.z);
      mpFilter_->template addUpdateMeas<1>(poseUpdateMeas_, transform->header.stamp.toSec() + mpPoseUpdate_->timeOffset_);
      updateAndPublish();
    }
  }

  /** \brief ROS service handler for resetting the filter.
   */
  bool resetServiceCallback(std_srvs::Empty::Request & /*request*/,
                            std_srvs::Empty::Response & /*response*/)
  {
    requestReset();
    return true;
  }

  /** \brief ROS service handler for resetting the filter to a given pose.
   */
  bool resetToPoseServiceCallback(rovio::SrvResetToPose::Request &request,
                                  rovio::SrvResetToPose::Response & /*response*/)
  {
    V3D WrWM(request.T_IW.position.x, request.T_IW.position.y,
             request.T_IW.position.z);
    QPD qMW(request.T_IW.orientation.w, request.T_IW.orientation.x,
            request.T_IW.orientation.y, request.T_IW.orientation.z);
    requestResetToPose(WrWM, qMW);
    return true;
  }

  /** \brief Reset the filter when the next IMU measurement is received.
   *         The orientaetion is initialized using an accel. measurement.
   */
  void requestReset()
  {
    std::lock_guard<std::mutex> lock(m_filter_);
    if (!init_state_.isInitialized())
    {
      std::cout << "Reinitialization already triggered. Ignoring request...";
      return;
    }

    init_state_.state_ = FilterInitializationState::State::WaitForInitUsingAccel;
  }

  /** \brief Reset the filter when the next IMU measurement is received.
   *         The pose is initialized to the passed pose.
   *  @param WrWM - Position Vector, pointing from the World-Frame to the IMU-Frame, expressed in World-Coordinates.
   *  @param qMW  - Quaternion, expressing World-Frame in IMU-Coordinates (World Coordinates->IMU Coordinates)
   */
  void requestResetToPose(const V3D &WrWM, const QPD &qMW)
  {
    std::lock_guard<std::mutex> lock(m_filter_);
    if (!init_state_.isInitialized())
    {
      std::cout << "Reinitialization already triggered. Ignoring request...";
      return;
    }

    init_state_.WrWM_ = WrWM;
    init_state_.qMW_ = qMW;
    init_state_.state_ = FilterInitializationState::State::WaitForInitExternalPose;
  }

  /** \brief Executes the update step of the filter and publishes the updated data.
   */
  void updateAndPublish()
  {
    if (init_state_.isInitialized())
    {
      // Execute the filter update.
      const double t1 = (double)cv::getTickCount();
      int c1 = std::get<0>(mpFilter_->updateTimelineTuple_).measMap_.size();
      static double timing_T = 0;
      static int timing_C = 0;
      const double oldSafeTime = mpFilter_->safe_.t_;
      mpFilter_->updateSafe();
      const double t2 = (double)cv::getTickCount();
      int c2 = std::get<0>(mpFilter_->updateTimelineTuple_).measMap_.size();
      timing_T += (t2 - t1) / cv::getTickFrequency() * 1000;
      timing_C += c1 - c2;
      bool plotTiming = false;
      if (plotTiming)
      {
        ROS_INFO_STREAM(" == Filter Update: " << (t2 - t1) / cv::getTickFrequency() * 1000 << " ms for processing " << c1 - c2 << " images, average: " << timing_T / timing_C);
      }
      if (mpFilter_->safe_.t_ > oldSafeTime)
      { // Publish only if something changed
        for (int i = 0; i < mtState::nCam_; i++)
        {
          if (!mpFilter_->safe_.img_[i].empty() && mpImgUpdate_->publishVisualisedFrame_)
          {
            std_msgs::Header header;
            header.seq = msgSeq_;
            header.stamp = ros::Time(mpFilter_->safe_.t_);
            image_pub_.publish(cv_bridge::CvImage(header, "bgr8", mpFilter_->safe_.img_[i]).toImageMsg());
          }
          if (!mpFilter_->safe_.img_[i].empty() && mpImgUpdate_->doFrameVisualisation_)
          {
            cv::imshow("Tracker" + std::to_string(i), mpFilter_->safe_.img_[i]);
            cv::waitKey(3);
          }
        }
        if (!mpFilter_->safe_.patchDrawing_.empty() && mpImgUpdate_->visualizePatches_)
        {
          cv::imshow("Patches", mpFilter_->safe_.patchDrawing_);
          cv::waitKey(3);
        }

        // Obtain the save filter state.
        mtFilterState &filterState = mpFilter_->safe_;
        mtState &state = mpFilter_->safe_.state_;
        state.updateMultiCameraExtrinsics(&mpFilter_->multiCamera_);
        MXD &cov = mpFilter_->safe_.cov_;
        imuOutputCT_.transformState(state, imuOutput_);

        if (mpFilter_->bspFilter_)
        {
          // Send IMU pose.
          tf::StampedTransform tf_transform_MW;
          tf_transform_MW.frame_id_ = world_frame_;
          tf_transform_MW.child_frame_id_ = imu_frame_ + "_bsp";
          tf_transform_MW.stamp_ = ros::Time::now();
          tf_transform_MW.setOrigin(tf::Vector3(imuOutput_.WrWB()(0), imuOutput_.WrWB()(1), imuOutput_.WrWB()(2)));
          tf_transform_MW.setRotation(tf::Quaternion(imuOutput_.qBW().x(), imuOutput_.qBW().y(), imuOutput_.qBW().z(), imuOutput_.qBW().w()));
          tb_.sendTransform(tf_transform_MW);

          // Send camera pose.
          for (int camID = 0; camID < mtState::nCam_; camID++)
          {
            tf::StampedTransform tf_transform_CM;
            tf_transform_CM.frame_id_ = imu_frame_ + "_bsp";
            tf_transform_CM.child_frame_id_ = camera_frame_ + std::to_string(camID) + "_bsp";
            tf_transform_CM.stamp_ = ros::Time::now();
            tf_transform_CM.setOrigin(tf::Vector3(state.MrMC(camID)(0), state.MrMC(camID)(1), state.MrMC(camID)(2)));
            tf_transform_CM.setRotation(tf::Quaternion(state.qCM(camID).x(), state.qCM(camID).y(), state.qCM(camID).z(), state.qCM(camID).w()));
            tb_.sendTransform(tf_transform_CM);
          }
        }
        else
        {
          // Cout verbose for pose measurements
          if (mpImgUpdate_->verbose_)
          {
            if (mpPoseUpdate_->inertialPoseIndex_ >= 0)
            {
              std::cout << "Transformation between inertial frames, IrIW, qWI: " << std::endl;
              std::cout << "  " << state.poseLin(mpPoseUpdate_->inertialPoseIndex_).transpose() << std::endl;
              std::cout << "  " << state.poseRot(mpPoseUpdate_->inertialPoseIndex_) << std::endl;
            }
            if (mpPoseUpdate_->bodyPoseIndex_ >= 0)
            {
              std::cout << "Transformation between body frames, MrMV, qVM: " << std::endl;
              std::cout << "  " << state.poseLin(mpPoseUpdate_->bodyPoseIndex_).transpose() << std::endl;
              std::cout << "  " << state.poseRot(mpPoseUpdate_->bodyPoseIndex_) << std::endl;
            }
          }

          // Send Map (Pose Sensor, I) to World (rovio-intern, W) transformation
          if (mpPoseUpdate_->inertialPoseIndex_ >= 0)
          {
            Eigen::Vector3d IrIW = state.poseLin(mpPoseUpdate_->inertialPoseIndex_);
            rot::RotationQuaternionPD qWI = state.poseRot(mpPoseUpdate_->inertialPoseIndex_);
            tf::StampedTransform tf_transform_WI;
            tf_transform_WI.frame_id_ = map_frame_;
            tf_transform_WI.child_frame_id_ = world_frame_;
            tf_transform_WI.stamp_ = ros::Time(mpFilter_->safe_.t_);
            tf_transform_WI.setOrigin(tf::Vector3(IrIW(0), IrIW(1), IrIW(2)));
            tf_transform_WI.setRotation(tf::Quaternion(qWI.x(), qWI.y(), qWI.z(), qWI.w()));
            tb_.sendTransform(tf_transform_WI);
          }

          // Send IMU pose.
          tf::StampedTransform tf_transform_MW;
          tf_transform_MW.frame_id_ = world_frame_;
          tf_transform_MW.child_frame_id_ = imu_frame_;
          tf_transform_MW.stamp_ = ros::Time(mpFilter_->safe_.t_);
          tf_transform_MW.setOrigin(tf::Vector3(imuOutput_.WrWB()(0), imuOutput_.WrWB()(1), imuOutput_.WrWB()(2)));
          tf_transform_MW.setRotation(tf::Quaternion(imuOutput_.qBW().x(), imuOutput_.qBW().y(), imuOutput_.qBW().z(), imuOutput_.qBW().w()));
          tb_.sendTransform(tf_transform_MW);

          // Send camera pose.
          for (int camID = 0; camID < mtState::nCam_; camID++)
          {
            tf::StampedTransform tf_transform_CM;
            tf_transform_CM.frame_id_ = imu_frame_;
            tf_transform_CM.child_frame_id_ = camera_frame_ + std::to_string(camID);
            tf_transform_CM.stamp_ = ros::Time(mpFilter_->safe_.t_);
            tf_transform_CM.setOrigin(tf::Vector3(state.MrMC(camID)(0), state.MrMC(camID)(1), state.MrMC(camID)(2)));
            tf_transform_CM.setRotation(tf::Quaternion(state.qCM(camID).x(), state.qCM(camID).y(), state.qCM(camID).z(), state.qCM(camID).w()));
            tb_.sendTransform(tf_transform_CM);
          }
        }

        // Publish Odometry
        if (pubOdometry_.getNumSubscribers() > 0 || forceOdometryPublishing_)
        {
          // Compute covariance of output
          imuOutputCT_.transformCovMat(state, cov, imuOutputCov_);

          odometryMsg_.header.seq = msgSeq_;
          odometryMsg_.header.stamp = ros::Time(mpFilter_->safe_.t_);
          odometryMsg_.pose.pose.position.x = imuOutput_.WrWB()(0);
          odometryMsg_.pose.pose.position.y = imuOutput_.WrWB()(1);
          odometryMsg_.pose.pose.position.z = imuOutput_.WrWB()(2);
          odometryMsg_.pose.pose.orientation.w = imuOutput_.qBW().w();
          odometryMsg_.pose.pose.orientation.x = imuOutput_.qBW().x();
          odometryMsg_.pose.pose.orientation.y = imuOutput_.qBW().y();
          odometryMsg_.pose.pose.orientation.z = imuOutput_.qBW().z();
          for (unsigned int i = 0; i < 6; i++)
          {
            unsigned int ind1 = mtOutput::template getId<mtOutput::_pos>() + i;
            if (i >= 3)
              ind1 = mtOutput::template getId<mtOutput::_att>() + i - 3;
            for (unsigned int j = 0; j < 6; j++)
            {
              unsigned int ind2 = mtOutput::template getId<mtOutput::_pos>() + j;
              if (j >= 3)
                ind2 = mtOutput::template getId<mtOutput::_att>() + j - 3;
              odometryMsg_.pose.covariance[j + 6 * i] = imuOutputCov_(ind1, ind2);
            }
          }
          odometryMsg_.twist.twist.linear.x = imuOutput_.BvB()(0);
          odometryMsg_.twist.twist.linear.y = imuOutput_.BvB()(1);
          odometryMsg_.twist.twist.linear.z = imuOutput_.BvB()(2);
          odometryMsg_.twist.twist.angular.x = imuOutput_.BwWB()(0);
          odometryMsg_.twist.twist.angular.y = imuOutput_.BwWB()(1);
          odometryMsg_.twist.twist.angular.z = imuOutput_.BwWB()(2);
          for (unsigned int i = 0; i < 6; i++)
          {
            unsigned int ind1 = mtOutput::template getId<mtOutput::_vel>() + i;
            if (i >= 3)
              ind1 = mtOutput::template getId<mtOutput::_ror>() + i - 3;
            for (unsigned int j = 0; j < 6; j++)
            {
              unsigned int ind2 = mtOutput::template getId<mtOutput::_vel>() + j;
              if (j >= 3)
                ind2 = mtOutput::template getId<mtOutput::_ror>() + j - 3;
              odometryMsg_.twist.covariance[j + 6 * i] = imuOutputCov_(ind1, ind2);
            }
          }
          pubOdometry_.publish(odometryMsg_);
        }

        // Send IMU pose message.
        if (pubTransform_.getNumSubscribers() > 0 || forceTransformPublishing_)
        {
          transformMsg_.header.seq = msgSeq_;
          transformMsg_.header.stamp = ros::Time(mpFilter_->safe_.t_);
          transformMsg_.transform.translation.x = imuOutput_.WrWB()(0);
          transformMsg_.transform.translation.y = imuOutput_.WrWB()(1);
          transformMsg_.transform.translation.z = imuOutput_.WrWB()(2);
          transformMsg_.transform.rotation.x = imuOutput_.qBW().x();
          transformMsg_.transform.rotation.y = imuOutput_.qBW().y();
          transformMsg_.transform.rotation.z = imuOutput_.qBW().z();
          transformMsg_.transform.rotation.w = imuOutput_.qBW().w();
          pubTransform_.publish(transformMsg_);
        }

        // Publish Extrinsics
        for (int camID = 0; camID < mtState::nCam_; camID++)
        {
          if (pubExtrinsics_[camID].getNumSubscribers() > 0 || forceExtrinsicsPublishing_)
          {
            extrinsicsMsg_[camID].header.seq = msgSeq_;
            extrinsicsMsg_[camID].header.stamp = ros::Time(mpFilter_->safe_.t_);
            extrinsicsMsg_[camID].pose.pose.position.x = state.MrMC(camID)(0);
            extrinsicsMsg_[camID].pose.pose.position.y = state.MrMC(camID)(1);
            extrinsicsMsg_[camID].pose.pose.position.z = state.MrMC(camID)(2);
            extrinsicsMsg_[camID].pose.pose.orientation.x = state.qCM(camID).x();
            extrinsicsMsg_[camID].pose.pose.orientation.y = state.qCM(camID).y();
            extrinsicsMsg_[camID].pose.pose.orientation.z = state.qCM(camID).z();
            extrinsicsMsg_[camID].pose.pose.orientation.w = state.qCM(camID).w();
            for (unsigned int i = 0; i < 6; i++)
            {
              unsigned int ind1 = mtState::template getId<mtState::_vep>(camID) + i;
              if (i >= 3)
                ind1 = mtState::template getId<mtState::_vea>(camID) + i - 3;
              for (unsigned int j = 0; j < 6; j++)
              {
                unsigned int ind2 = mtState::template getId<mtState::_vep>(camID) + j;
                if (j >= 3)
                  ind2 = mtState::template getId<mtState::_vea>(camID) + j - 3;
                extrinsicsMsg_[camID].pose.covariance[j + 6 * i] = cov(ind1, ind2);
              }
            }
            pubExtrinsics_[camID].publish(extrinsicsMsg_[camID]);
          }
        }

        // Publish IMU biases
        if (pubImuBias_.getNumSubscribers() > 0 || forceImuBiasPublishing_)
        {
          imuBiasMsg_.header.seq = msgSeq_;
          imuBiasMsg_.header.stamp = ros::Time(mpFilter_->safe_.t_);
          imuBiasMsg_.angular_velocity.x = state.gyb()(0);
          imuBiasMsg_.angular_velocity.y = state.gyb()(1);
          imuBiasMsg_.angular_velocity.z = state.gyb()(2);
          imuBiasMsg_.linear_acceleration.x = state.acb()(0);
          imuBiasMsg_.linear_acceleration.y = state.acb()(1);
          imuBiasMsg_.linear_acceleration.z = state.acb()(2);
          for (int i = 0; i < 3; i++)
          {
            for (int j = 0; j < 3; j++)
            {
              imuBiasMsg_.angular_velocity_covariance[3 * i + j] = cov(mtState::template getId<mtState::_gyb>() + i, mtState::template getId<mtState::_gyb>() + j);
            }
          }
          for (int i = 0; i < 3; i++)
          {
            for (int j = 0; j < 3; j++)
            {
              imuBiasMsg_.linear_acceleration_covariance[3 * i + j] = cov(mtState::template getId<mtState::_acb>() + i, mtState::template getId<mtState::_acb>() + j);
            }
          }
          pubImuBias_.publish(imuBiasMsg_);
        }

        // PointCloud message.
        if (pubPcl_.getNumSubscribers() > 0 || pubMarkers_.getNumSubscribers() > 0 || forcePclPublishing_ || forceMarkersPublishing_)
        {
          pclMsg_.header.seq = msgSeq_;
          pclMsg_.header.stamp = ros::Time(mpFilter_->safe_.t_);
          markerMsg_.header.seq = msgSeq_;
          markerMsg_.header.stamp = ros::Time(mpFilter_->safe_.t_);
          markerMsg_.points.clear();
          float badPoint = std::numeric_limits<float>::quiet_NaN(); // Invalid point.
          int offset = 0;

          FeatureDistance distance;
          double d, d_minus, d_plus;
          const double stretchFactor = 3;
          for (unsigned int i = 0; i < mtState::nMax_; i++, offset += pclMsg_.point_step)
          {
            if (filterState.fsm_.isValid_[i])
            {
              // Get 3D feature coordinates.
              int camID = filterState.fsm_.features_[i].mpCoordinates_->camID_;
              distance = state.dep(i);
              d = distance.getDistance();
              const double sigma = sqrt(cov(mtState::template getId<mtState::_fea>(i) + 2, mtState::template getId<mtState::_fea>(i) + 2));
              distance.p_ -= stretchFactor * sigma;
              d_minus = distance.getDistance();
              if (d_minus > 1000)
                d_minus = 1000;
              if (d_minus < 0)
                d_minus = 0;
              distance.p_ += 2 * stretchFactor * sigma;
              d_plus = distance.getDistance();
              if (d_plus > 1000)
                d_plus = 1000;
              if (d_plus < 0)
                d_plus = 0;
              Eigen::Vector3d bearingVector = filterState.state_.CfP(i).get_nor().getVec();
              const Eigen::Vector3d CrCPm = bearingVector * d_minus;
              const Eigen::Vector3d CrCPp = bearingVector * d_plus;
              const Eigen::Vector3f MrMPm = V3D(mpFilter_->multiCamera_.BrBC_[camID] + mpFilter_->multiCamera_.qCB_[camID].inverseRotate(CrCPm)).cast<float>();
              const Eigen::Vector3f MrMPp = V3D(mpFilter_->multiCamera_.BrBC_[camID] + mpFilter_->multiCamera_.qCB_[camID].inverseRotate(CrCPp)).cast<float>();

              // Get human readable output
              transformFeatureOutputCT_.setFeatureID(i);
              transformFeatureOutputCT_.setOutputCameraID(filterState.fsm_.features_[i].mpCoordinates_->camID_);
              transformFeatureOutputCT_.transformState(state, featureOutput_);
              transformFeatureOutputCT_.transformCovMat(state, cov, featureOutputCov_);
              featureOutputReadableCT_.transformState(featureOutput_, featureOutputReadable_);
              featureOutputReadableCT_.transformCovMat(featureOutput_, featureOutputCov_, featureOutputReadableCov_);

              // Get landmark output
              landmarkOutputImuCT_.setFeatureID(i);
              landmarkOutputImuCT_.transformState(state, landmarkOutput_);
              landmarkOutputImuCT_.transformCovMat(state, cov, landmarkOutputCov_);
              const Eigen::Vector3f MrMP = landmarkOutput_.get<LandmarkOutput::_lmk>().template cast<float>();

              // Write feature id, camera id, and rgb
              uint8_t gray = 255;
              uint32_t rgb = (gray << 16) | (gray << 8) | gray;
              uint32_t status = filterState.fsm_.features_[i].mpStatistics_->status_[0];
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[0].offset], &filterState.fsm_.features_[i].idx_, sizeof(int)); // id
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[1].offset], &camID, sizeof(int));                              // cam id
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[2].offset], &rgb, sizeof(uint32_t));                           // rgb
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[3].offset], &status, sizeof(int));                             // status

              // Write coordinates to pcl message.
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[4].offset], &MrMP[0], sizeof(float)); // x
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[5].offset], &MrMP[1], sizeof(float)); // y
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[6].offset], &MrMP[2], sizeof(float)); // z

              // Add feature bearing vector and distance
              const Eigen::Vector3f bearing = featureOutputReadable_.bea().template cast<float>();
              const float distance = static_cast<float>(featureOutputReadable_.dis());
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[7].offset], &bearing[0], sizeof(float)); // x
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[8].offset], &bearing[1], sizeof(float)); // y
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[9].offset], &bearing[2], sizeof(float)); // z
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[10].offset], &distance, sizeof(float));  // d

              // Add the corresponding covariance (upper triangular)
              Eigen::Matrix3f cov_MrMP = landmarkOutputCov_.cast<float>();
              int mCounter = 11;
              for (int row = 0; row < 3; row++)
              {
                for (int col = row; col < 3; col++)
                {
                  memcpy(&pclMsg_.data[offset + pclMsg_.fields[mCounter].offset], &cov_MrMP(row, col), sizeof(float));
                  mCounter++;
                }
              }

              // Add distance uncertainty
              const float distance_cov = static_cast<float>(featureOutputReadableCov_(3, 3));
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[mCounter].offset], &distance_cov, sizeof(float));

              // Line markers (Uncertainty rays).
              geometry_msgs::Point point_near_msg;
              geometry_msgs::Point point_far_msg;
              point_near_msg.x = float(CrCPp[0]);
              point_near_msg.y = float(CrCPp[1]);
              point_near_msg.z = float(CrCPp[2]);
              point_far_msg.x = float(CrCPm[0]);
              point_far_msg.y = float(CrCPm[1]);
              point_far_msg.z = float(CrCPm[2]);
              markerMsg_.points.push_back(point_near_msg);
              markerMsg_.points.push_back(point_far_msg);
            }
            else
            {
              // If current feature is not valid copy NaN
              int id = -1;
              memcpy(&pclMsg_.data[offset + pclMsg_.fields[0].offset], &id, sizeof(int)); // id
              for (int j = 1; j < pclMsg_.fields.size(); j++)
              {
                memcpy(&pclMsg_.data[offset + pclMsg_.fields[j].offset], &badPoint, sizeof(float));
              }
            }
          }
          pubPcl_.publish(pclMsg_);
          pubMarkers_.publish(markerMsg_);
        }
        if (pubPatch_.getNumSubscribers() > 0 || forcePatchPublishing_)
        {
          patchMsg_.header.seq = msgSeq_;
          patchMsg_.header.stamp = ros::Time(mpFilter_->safe_.t_);
          int offset = 0;
          for (unsigned int i = 0; i < mtState::nMax_; i++, offset += patchMsg_.point_step)
          {
            if (filterState.fsm_.isValid_[i])
            {
              memcpy(&patchMsg_.data[offset + patchMsg_.fields[0].offset], &filterState.fsm_.features_[i].idx_, sizeof(int)); // id
              // Add patch data
              for (int l = 0; l < mtState::nLevels_; l++)
              {
                for (int y = 0; y < mtState::patchSize_; y++)
                {
                  for (int x = 0; x < mtState::patchSize_; x++)
                  {
                    memcpy(&patchMsg_.data[offset + patchMsg_.fields[1].offset + (l * mtState::patchSize_ * mtState::patchSize_ + y * mtState::patchSize_ + x) * 4], &filterState.fsm_.features_[i].mpMultilevelPatch_->patches_[l].patch_[y * mtState::patchSize_ + x], sizeof(float)); // Patch
                    memcpy(&patchMsg_.data[offset + patchMsg_.fields[2].offset + (l * mtState::patchSize_ * mtState::patchSize_ + y * mtState::patchSize_ + x) * 4], &filterState.fsm_.features_[i].mpMultilevelPatch_->patches_[l].dx_[y * mtState::patchSize_ + x], sizeof(float));    // dx
                    memcpy(&patchMsg_.data[offset + patchMsg_.fields[3].offset + (l * mtState::patchSize_ * mtState::patchSize_ + y * mtState::patchSize_ + x) * 4], &filterState.fsm_.features_[i].mpMultilevelPatch_->patches_[l].dy_[y * mtState::patchSize_ + x], sizeof(float));    // dy
                    memcpy(&patchMsg_.data[offset + patchMsg_.fields[4].offset + (l * mtState::patchSize_ * mtState::patchSize_ + y * mtState::patchSize_ + x) * 4], &filterState.mlpErrorLog_[i].patches_[l].patch_[y * mtState::patchSize_ + x], sizeof(float));                       // error
                  }
                }
              }
            }
            else
            {
              // If current feature is not valid copy NaN
              int id = -1;
              memcpy(&patchMsg_.data[offset + patchMsg_.fields[0].offset], &id, sizeof(int)); // id
            }
          }

          pubPatch_.publish(patchMsg_);
        }

        // Bsp: update data respective to belief (filter) state
        for (unsigned int i = 0; i < mtState::nMax_; ++i)
          mpFilter_->bsp_featureParams_[i].getParams(i, filterState, mpFilter_->octree_);
        // Bsp: visualization extras
        if (BSP_pubFrustum_.getNumSubscribers() > 0)
        {
          BSP_frustumMsg_.header.seq = msgSeq_;
          if (mpFilter_->bspFilter_)
            BSP_frustumMsg_.header.stamp = ros::Time::now();
          else
            BSP_frustumMsg_.header.stamp = ros::Time(filterState.t_);
          geometry_msgs::Point line_point;
          BSP_frustumMsg_.points.clear();
          for (unsigned int i = 0; i < mt_BspFeatureParams::cam_numPlanes * mt_BspFeatureParams::cam_numPointsPerPlane; ++i)
          {
            const V3D &cam_frustCWP_i = mt_BspFeatureParams::cam_frustCW[i];
            line_point.x = cam_frustCWP_i.x();
            line_point.y = cam_frustCWP_i.y();
            line_point.z = cam_frustCWP_i.z();
            BSP_frustumMsg_.points.push_back(line_point);
          }
          BSP_pubFrustum_.publish(BSP_frustumMsg_);
        }
        if (BSP_pubBearingArrows_.getNumSubscribers() > 0)
        {
          for (unsigned int i = 0; i < mtState::nMax_; ++i)
          {
            visualization_msgs::Marker &BSP_bearingArrow_i = BSP_bearingArrowArrayMsg_.markers[i];
            BSP_bearingArrow_i.header.seq = msgSeq_;
            if (mpFilter_->bspFilter_)
            {
              BSP_bearingArrow_i.header.stamp = ros::Time::now();
              BSP_bearingArrow_i.header.frame_id = "camera" + std::to_string(state.CfP(i).camID_) + "_bsp";
            }
            else
            {
              BSP_bearingArrow_i.header.stamp = ros::Time(filterState.t_);
              BSP_bearingArrow_i.header.frame_id = "camera" + std::to_string(state.CfP(i).camID_);
            }
            if (filterState.fsm_.isValid_[i])
            {
              // Calculate landmark features in world frame
              Eigen::Vector3d CrCP_i = state.dep(i).getDistance() * state.CfP(i).get_nor().getVec();
              Eigen::Vector3d MrMP_i = state.MrMC(state.CfP(i).camID_) + state.qCM(state.CfP(i).camID_).inverseRotate(CrCP_i);
              Eigen::Vector3d fea_params_i = state.WrWM() + state.qWM().rotate(MrMP_i);
              Eigen::Vector3d T_WtoC_i = state.template get<mtState::_pos>() + state.template get<mtState::_att>().rotate(state.MrMC(state.CfP(i).camID_));
              QPD qCW_i = state.qCM(state.CfP(i).camID_) * (state.template get<mtState::_att>().inverted());
              Eigen::Quaterniond R_CtoW_i(qCW_i.w(), qCW_i.x(), qCW_i.y(), qCW_i.z());
              Eigen::Vector3d fea_C_i = R_CtoW_i.inverse() * (fea_params_i - T_WtoC_i);
              // calculate depth-bearing params based off world frame
              BSP_bearingArrow_i.pose.position.x = BSP_bearingArrow_i.pose.position.y = BSP_bearingArrow_i.pose.position.z = 0;
              static const Eigen::Vector3d init(1.0, 0.0, 0.0);
              Eigen::Vector3d dir(fea_C_i.x(), fea_C_i.y(), fea_C_i.z());
              Eigen::Quaterniond bearing_quat;
              bearing_quat.setFromTwoVectors(init, dir);
              bearing_quat.normalize();
              BSP_bearingArrow_i.pose.orientation.x = bearing_quat.x();
              BSP_bearingArrow_i.pose.orientation.y = bearing_quat.y();
              BSP_bearingArrow_i.pose.orientation.z = bearing_quat.z();
              BSP_bearingArrow_i.pose.orientation.w = bearing_quat.w();
              BSP_bearingArrow_i.scale.x = dir.norm();
              if (!mpFilter_->bsp_featureParams_[i].bsp_fov())
              {
                BSP_bearingArrow_i.color.a = 0.5;
                BSP_bearingArrow_i.color.r = 0.25;
                BSP_bearingArrow_i.color.g = 0.25;
                BSP_bearingArrow_i.color.b = 0.25;
              }
              else if (mpFilter_->octree_ != nullptr)
              {
                unsigned int bsp_los = mpFilter_->bsp_featureParams_[i].bsp_los();
                if (bsp_los == bsp::LoS::Free)
                {
                  BSP_bearingArrow_i.color.a = 1.0;
                  BSP_bearingArrow_i.color.r = 0.0;
                  BSP_bearingArrow_i.color.g = 1.0;
                  BSP_bearingArrow_i.color.b = 0.0;
                }
                else
                {
                  BSP_bearingArrow_i.color.a = 1.0;
                  BSP_bearingArrow_i.color.r = (bsp_los % (10 * bsp::LoS::Occupied)) / bsp::LoS::Occupied;
                  BSP_bearingArrow_i.color.g = 0.0;
                  BSP_bearingArrow_i.color.b = (bsp_los % (10 * bsp::LoS::Unknown)) / bsp::LoS::Unknown;
                }
              }
              else
              {
                BSP_bearingArrow_i.color.a = 1.0;
                BSP_bearingArrow_i.color.r = 0.0;
                BSP_bearingArrow_i.color.g = 1.0;
                BSP_bearingArrow_i.color.b = 1.0;
              }
            }
            else
            {
              BSP_bearingArrow_i.color.a = 0.0;
            }
          }
          BSP_pubBearingArrows_.publish(BSP_bearingArrowArrayMsg_);
        }

        gotFirstMessages_ = true;
      }
    }
  }
};

} // namespace rovio

#endif /* ROVIO_ROVIONODE_HPP_ */
