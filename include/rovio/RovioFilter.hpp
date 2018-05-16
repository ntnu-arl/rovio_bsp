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

#ifndef ROVIO_ROVIO_FILTER_HPP_
#define ROVIO_ROVIO_FILTER_HPP_

#include "lightweight_filtering/common.hpp"
#include "lightweight_filtering/FilterBase.hpp"
#include "rovio/FilterStates.hpp"
#include "rovio/ImgUpdate.hpp"
#include "rovio/PoseUpdate.hpp"
#include "rovio/ImuPrediction.hpp"
#include "rovio/MultiCamera.hpp"

/*
 * Bsp: extras to pack-&-send and receive belief (filter) state
 */
#include <bsp_msgs/ROVIO_Point2fMsg.h>
#include <bsp_msgs/ROVIO_RobocentricFeatureElementMsg.h>
#include <bsp_msgs/ROVIO_StateMsg.h>
#include <bsp_msgs/ROVIO_FilterStateMsg.h>
#include <bsp_msgs/BSP_TrajectoryReferenceMsg.h>
#include <bsp_msgs/BSP_SrvSendFilterState.h>
#include <bsp_msgs/BSP_SrvPropagateFilterState.h>
#include <eigen_conversions/eigen_msg.h>

/*
 * Bsp: extras to check feature line-of-sight visibility during propagation
 */
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/GetOctomap.h>
#define SYMBOL_INIT(scope,symbol) decltype(scope::symbol) scope::symbol = {}
#define SYMBOL_ALIAS_INIT(scope,symbol) decltype(scope::symbol)& symbol = scope::symbol

namespace bsp {
  void msgToMatrixEigen(std_msgs::Float64MultiArray &m, MXD &e)  
  {
    typename MXD::Index rows = m.layout.dim[0].size;
    typename MXD::Index cols = m.layout.dim[1].stride;
    if (m.layout.dim.size()!=2 || m.layout.dim[1].size!=cols || m.layout.dim[0].stride!=rows*cols)
      throw std::runtime_error("msgToMatrixEigen: Incompatible sizes of std_msgs::Float64MultiArray message and std_msgs::Float64MultiArray eigen matrix...");
    e = typename Eigen::Map<MXD>( &(m.data)[0], rows, cols );
  }

  enum LoS { Free = 0, Unknown = 1, Occupied = 10 };
  unsigned int getLoS(const octomap::OcTree& octree, const V3D& start, const V3D& end, unsigned int continue_thres=1) {
    octomap::KeyRay key_ray;
    octree.computeRayKeys(octomap::point3d(start.x(),start.y(),start.z()), octomap::point3d(end.x(),end.y(),end.z()), key_ray);
    const octomap::OcTreeKey& end_key = octree.coordToKey(octomap::point3d(end.x(),end.y(),end.z()));
    continue_thres = std::min(continue_thres, static_cast<unsigned int>(LoS::Unknown));
    unsigned int result = LoS::Free;
    for (octomap::OcTreeKey key : key_ray) {
      if (key != end_key) {
        octomap::OcTreeNode* node = octree.search(key);
        if (node == nullptr) {
	  result += LoS::Unknown;
	  if (!--continue_thres){ break; }
        } else if (octree.isNodeOccupied(node)) {
          result += LoS::Occupied;
	  if (!--continue_thres){ break; }
        }
      }
    }
    if (continue_thres)
      return LoS::Free;
    else
      return result;
  }
}

namespace rot = kindr::rotations::eigen_impl;

namespace rovio {

/** \brief Bsp: class, wrapping a std::tuple used for Bsp feature parameters container
 */ 
typedef std::tuple<bool, cv::Point2d, V3D, double, V3D, M3D , bool , unsigned int> BSP_FeatureTuple;
template<typename DERIVED_BASE>
class BSP_FeatureParams : public BSP_FeatureTuple {

  public:
  
    typedef typename DERIVED_BASE::mtFilterState mtFilterState;
    typedef typename DERIVED_BASE::mtState mtState;

    static constexpr unsigned int _bsp_val = 0;			/**<Valid feature flag (bool).*/
    static constexpr unsigned int _bsp_p2D = _bsp_val+1;	/**<2D Pixel coordinates of feature (cv::Point2d).*/
    static constexpr unsigned int _bsp_b3D = _bsp_p2D+1;	/**<3D Bearing vector of robocentric feature (Eigen::Vector3d).*/
    static constexpr unsigned int _bsp_dep = _bsp_b3D+1;	/**<Depth of robocentric feature (double).*/
    static constexpr unsigned int _bsp_w3D = _bsp_dep+1;	/**<3D World coordinates of feature (Eigen::Vector3d).*/
    static constexpr unsigned int _bsp_cov = _bsp_w3D+1;	/**<3x3 Covariance of bearing, depth parametrization of feature (Eigen::Matrix3d).*/
    static constexpr unsigned int _bsp_fov = _bsp_cov+1;	/**<In FoV (field-of-view of respective camera) feature flag (bool).*/
    static constexpr unsigned int _bsp_los = _bsp_fov+1;	/**<Los (line-of-sight depending on octomap ray-casting) feature status (unsigned int).*/
      static constexpr MXD::Index triplet_ = 3;
      static constexpr MXD::Index sixlet_ = 6;
    static constexpr MXD::Index pos_idx_ = triplet_*mtState::_pos; 			/**<Index of pos.*/
    static constexpr MXD::Index vel_idx_ = triplet_*mtState::_vel; 			/**<Index of vel.*/
    static constexpr MXD::Index acb_idx_ = triplet_*mtState::_acb; 			/**<Index of acb.*/
    static constexpr MXD::Index gyb_idx_ = triplet_*mtState::_gyb; 			/**<Index of gyb.*/
    static constexpr MXD::Index att_idx_ = triplet_*mtState::_att; 			/**<Index of att.*/
      static constexpr MXD::Index yaw_idx_ = triplet_*mtState::_att+2;			/**<Index of yaw, used in yawOnly_ (special) case (exclusive via static_assert).*/
    static constexpr MXD::Index vep_0_idx_ = triplet_*mtState::_vep; 			/**<Index of first camera translation extrinsics.*/
    static constexpr MXD::Index vea_0_idx_ = triplet_*mtState::_vea;  			/**<Index of first camera rotation extrinsics.*/
    static constexpr MXD::Index fea_0_idx_ = vep_0_idx_ + sixlet_*mtState::nCam_; 	/**<Index of first feature.*/
      static constexpr MXD::Index dep_0_idx_ = fea_0_idx_+2;				/**<Index of first feature depth, used in depthOnly_ (special) case (exclusive via static_assert).*/
    static constexpr MXD::Index covMat_size_ = vep_0_idx_ + sixlet_*mtState::nCam_ + triplet_*mtState::nMax_;	/**<Size of covMat.*/
    
    static int map_featureVisibilityThreshold_;	/**<Line-of-Sight visibility threshold (static class property, int only to to allow parsing from Base intRegister).*/
    static double cam_FoVx_;			/**<Camera Field-of-View radial width (static class property).*/ 
    static double cam_FoVy_;			/**<Camera Field-of-View radial height (static class property).*/ 
    static double cam_FoVz_;			/**<Camera Field-of-View depth (static class property).*/ 

    static double cam_backW;
    static double cam_backH;
    static double cam_frontW;
    static double cam_frontH; 

    static constexpr double cam_planeBound = -1e-15;
    static constexpr unsigned int cam_numPlanes = 6;
    static constexpr unsigned int cam_numPointsPerPlane = 3;
    static V3D cam_frust[cam_numPlanes*cam_numPointsPerPlane];
    static V3D cam_frustCW[cam_numPlanes*cam_numPointsPerPlane];

  /** \brief Bsp: get<_bsp_xxx> alias method (avoid calls like get<mtFilter::mt_BspFeatureParams::_bsp_xxx>).*/
  const std::tuple_element<_bsp_val, BSP_FeatureTuple>::type & bsp_val(){  return std::get<_bsp_val>(*this);  }
  /** \brief Bsp: get<_bsp_xxx> alias method (avoid calls like get<mtFilter::mt_BspFeatureParams::_bsp_xxx>).*/
  const std::tuple_element<_bsp_p2D, BSP_FeatureTuple>::type & bsp_p2D(){  return std::get<_bsp_p2D>(*this);  }
  /** \brief Bsp: get<_bsp_xxx> alias method (avoid calls like get<mtFilter::mt_BspFeatureParams::_bsp_xxx>).*/
  const std::tuple_element<_bsp_b3D, BSP_FeatureTuple>::type & bsp_b3D(){  return std::get<_bsp_b3D>(*this);  }
  /** \brief Bsp: get<_bsp_xxx> alias method (avoid calls like get<mtFilter::mt_BspFeatureParams::_bsp_xxx>).*/
  const std::tuple_element<_bsp_dep, BSP_FeatureTuple>::type & bsp_dep(){  return std::get<_bsp_dep>(*this);  }
  /** \brief Bsp: get<_bsp_xxx> alias method (avoid calls like get<mtFilter::mt_BspFeatureParams::_bsp_xxx>).*/
  const std::tuple_element<_bsp_w3D, BSP_FeatureTuple>::type & bsp_w3D(){  return std::get<_bsp_w3D>(*this);  }
  /** \brief Bsp: get<_bsp_xxx> alias method (avoid calls like get<mtFilter::mt_BspFeatureParams::_bsp_xxx>).*/
  const std::tuple_element<_bsp_cov, BSP_FeatureTuple>::type & bsp_cov(){  return std::get<_bsp_cov>(*this);  }
  /** \brief Bsp: get<_bsp_xxx> alias method (avoid calls like get<mtFilter::mt_BspFeatureParams::_bsp_xxx>).*/
  const std::tuple_element<_bsp_fov, BSP_FeatureTuple>::type & bsp_fov(){  return std::get<_bsp_fov>(*this);  }
  /** \brief Bsp: get<_bsp_xxx> alias method (avoid calls like get<mtFilter::mt_BspFeatureParams::_bsp_xxx>).*/
  const std::tuple_element<_bsp_los, BSP_FeatureTuple>::type & bsp_los(){  return std::get<_bsp_los>(*this);  }

  /** \brief Bsp: refresh static properties - used for feature visibility checking
   */
  static void refreshProperties(){
    cam_backW = cam_FoVz_ * tan( (M_PI/180)*(cam_FoVx_/2) )*2;
    cam_backH = cam_FoVz_ * tan( (M_PI/180)*(cam_FoVy_/2) )*2;
    cam_frontW = 4.51/1000;  //apertureWidth (Aptina: 4.51mm)
    cam_frontH = 2.88/1000; //apertureHeight (Aptina: 2.88mm) 

    // TODO: drop front, back?
    cam_frust[0] = V3D(cam_frontW/2,-cam_frontH/2,0);  cam_frust[1] = V3D(-cam_frontW/2,-cam_frontH/2,0);  cam_frust[2] = V3D(-cam_frontW/2,cam_frontH/2,0);                     //front
    cam_frust[3] = V3D(0,0,0);  cam_frust[4] = V3D(-cam_backW/2,-cam_backH/2,cam_FoVz_);  cam_frust[5] = V3D(-cam_backW/2,cam_backH/2,cam_FoVz_);                                //left 
    cam_frust[6] = V3D(0,0,0);  cam_frust[7] = V3D(cam_backW/2,cam_backH/2,cam_FoVz_);  cam_frust[8] = V3D(cam_backW/2,-cam_backH/2,cam_FoVz_);                                  //right 
    cam_frust[9] = V3D(0,0,0);  cam_frust[10] = V3D(cam_backW/2,-cam_backH/2,cam_FoVz_);  cam_frust[11] = V3D(-cam_backW/2,-cam_backH/2,cam_FoVz_);                              //top
    cam_frust[12] = V3D(0,0,0);  cam_frust[13] = V3D(-cam_backW/2,cam_backH/2,cam_FoVz_);  cam_frust[14] = V3D(cam_backW/2,cam_backH/2,cam_FoVz_);                               //bottom
    cam_frust[15] = V3D(cam_backW/2,-cam_backH/2,cam_FoVz_);  cam_frust[16] = V3D(cam_backW/2,cam_backH/2,cam_FoVz_);  cam_frust[17] = V3D(-cam_backW/2,cam_backH/2,cam_FoVz_);  //back
  }

  /** \brief Bsp: constructor
   */
  BSP_FeatureParams(){
    static_assert(_bsp_los+1==std::tuple_size<BSP_FeatureTuple>::value,"Error with indices");
  }

  /** \brief Bsp: destructor
   */
  virtual ~BSP_FeatureParams(){};

  /** \brief Bsp: extracts from FilterState the data required to populate BSP_FeatureParams.
   *
   *  @param idx         - Feature index.
   *  @param filterState - FilterState.
   *  @param octree      - Octomap.
   */
  void getParams(const unsigned int& idx, const mtFilterState& filterState, const octomap::OcTree* const& octree){
    const mtState& state = filterState.state_;

    if (std::get<_bsp_val>(*this) = filterState.fsm_.isValid_[idx]){
      const FeatureDistance& state_dep = state.dep(idx);
      const FeatureCoordinates& state_CfP = state.CfP(idx);
      std::get<_bsp_p2D>(*this) = filterState.fsm_.features_[idx].mpCoordinates_->c_;
      std::get<_bsp_b3D>(*this) = state_CfP.get_nor().getVec();
      std::get<_bsp_dep>(*this) = state_dep.getDistance();
      const V3D CrCP = std::get<_bsp_dep>(*this) * state_CfP.get_nor().getVec();
      const V3D MrMP= state.MrMC(state_CfP.camID_) + state.qCM(state_CfP.camID_).inverseRotate(CrCP);    
      const V3D fea_WM = state.template get<mtState::_pos>()+state.template get<mtState::_att>().rotate(MrMP);
      const V3D cam_WM = state.template get<mtState::_pos>()+state.template get<mtState::_att>().rotate(state.MrMC(state_CfP.camID_));
      std::get<_bsp_w3D>(*this) = fea_WM;
      std::get<_bsp_cov>(*this) = filterState.cov_.block(fea_0_idx_+triplet_*idx,fea_0_idx_+triplet_*idx,triplet_,triplet_);
      std::get<_bsp_fov>(*this) = this->getFoV(filterState);
      // Line-of-Sight visibility, considers fea_i-3*sigma(depth)
      if (octree != nullptr){
        const double sigma = sqrt(filterState.cov_(mtState::template getId<mtState::_fea>(idx)+2,mtState::template getId<mtState::_fea>(idx)+2));
        FeatureDistance dep_m(state_dep);
	dep_m.p_ -= 3.0*sigma;
        double dep_m_distance = dep_m.getDistance();
        if(dep_m_distance<0) dep_m_distance = 0; else if(dep_m_distance>1000) dep_m_distance = 1000;
        FeatureDistance dep_p(state_dep);
	dep_p.p_ += 3.0*sigma;
        double dep_p_distance = dep_p.getDistance();
        if(dep_p_distance<0) dep_p_distance = 0; else if(dep_p_distance>1000) dep_p_distance = 1000;
	// Lower-Bound depends on depth parametrization type, use min()
	const V3D CrCP_l = std::min(dep_m_distance,dep_p_distance)*state_CfP.get_nor().getVec();
	const V3D MrMP_l= state.MrMC(state_CfP.camID_) + state.qCM(state_CfP.camID_).inverseRotate(CrCP_l);    
        const V3D fea_WM_l = state.template get<mtState::_pos>()+state.template get<mtState::_att>().rotate(MrMP_l);
	std::get<_bsp_los>(*this) = bsp::getLoS(*octree, cam_WM, fea_WM_l, static_cast<unsigned int>(map_featureVisibilityThreshold_));
      }
      else{
	std::get<_bsp_los>(*this) = bsp::LoS::Free;
      }
    }

  }

  /** \brief Bsp: check if a Feature is in Field-of-View of a camera.
   *
   *  @param filterState - FilterState.
   */
  bool getFoV(const mtFilterState& filterState){
    const mtState& state = filterState.state_;

    const V3D T_WtoC = state.template get<mtState::_pos>() + state.template get<mtState::_att>().rotate(state.MrMC(0));
    const QPD qCW = state.qCM(0) * (state.template get<mtState::_att>().inverted());        
    const M3D R_CtoW(Eigen::Quaterniond(qCW.w(), qCW.x(), qCW.y(), qCW.z()));

    for (unsigned int i=0; i<cam_numPlanes*cam_numPointsPerPlane; ++i){
      cam_frustCW[i] = R_CtoW * cam_frust[i] + T_WtoC;
    }

    std::vector<V3D> cam_frustNormals(cam_numPlanes);
    for (unsigned int i=0,offset=0; i<cam_numPlanes; ++i, offset+=cam_numPointsPerPlane){
      const V3D faceV1(cam_frustCW[offset+1]-cam_frustCW[offset]);
      const V3D faceV2(cam_frustCW[offset+2]-cam_frustCW[offset]);
      V3D faceNor(faceV1.cross(faceV2));
      //faceNor /= faceNor.norm();  //TODO: needed?
      cam_frustNormals[i] = faceNor;
    }  

    for (unsigned int i=0,offset=0; i<cam_numPlanes; ++i, offset+=cam_numPointsPerPlane){
      V3D ptNor = cam_frustCW[offset] - std::get<_bsp_w3D>(*this); 
      //ptNor /= ptNor.norm();  //TODO: needed?
      if ( ptNor.dot(cam_frustNormals[i]) < cam_planeBound )
        return false;
    }

    return true;
  }

};

/** \brief Bsp: instantiate static properties */
template<typename DERIVED_BASE> SYMBOL_INIT(BSP_FeatureParams<DERIVED_BASE>,map_featureVisibilityThreshold_);
/** \brief Bsp: instantiate static properties */
template<typename DERIVED_BASE> SYMBOL_INIT(BSP_FeatureParams<DERIVED_BASE>,cam_FoVx_);
/** \brief Bsp: instantiate static properties */
template<typename DERIVED_BASE> SYMBOL_INIT(BSP_FeatureParams<DERIVED_BASE>,cam_FoVy_);
/** \brief Bsp: instantiate static properties */
template<typename DERIVED_BASE> SYMBOL_INIT(BSP_FeatureParams<DERIVED_BASE>,cam_FoVz_);
/** \brief Bsp: instantiate static properties */
template<typename DERIVED_BASE> SYMBOL_INIT(BSP_FeatureParams<DERIVED_BASE>,cam_backW);
/** \brief Bsp: instantiate static properties */
template<typename DERIVED_BASE> SYMBOL_INIT(BSP_FeatureParams<DERIVED_BASE>,cam_backH);
/** \brief Bsp: instantiate static properties */
template<typename DERIVED_BASE> SYMBOL_INIT(BSP_FeatureParams<DERIVED_BASE>,cam_frontW);
/** \brief Bsp: instantiate static properties */
template<typename DERIVED_BASE> SYMBOL_INIT(BSP_FeatureParams<DERIVED_BASE>,cam_frontH);
/** \brief Bsp: instantiate static properties */
template<typename DERIVED_BASE> SYMBOL_INIT(BSP_FeatureParams<DERIVED_BASE>,cam_frust);
/** \brief Bsp: instantiate static properties */
template<typename DERIVED_BASE> SYMBOL_INIT(BSP_FeatureParams<DERIVED_BASE>,cam_frustCW);


/** \brief Class, defining the Rovio Filter.
 *
 *  @tparam FILTERSTATE - \ref rovio::FilterState
 */
template<typename FILTERSTATE, bool BSPFILTER=false>
class RovioFilter:public LWF::FilterBase<ImuPrediction<FILTERSTATE>,ImgUpdate<FILTERSTATE>,PoseUpdate<FILTERSTATE,(int)(FILTERSTATE::mtState::nPose_>0)-1,(int)(FILTERSTATE::mtState::nPose_>1)*2-1>>{
 public:
  typedef LWF::FilterBase<ImuPrediction<FILTERSTATE>,ImgUpdate<FILTERSTATE>,PoseUpdate<FILTERSTATE,(int)(FILTERSTATE::mtState::nPose_>0)-1,(int)(FILTERSTATE::mtState::nPose_>1)*2-1>> Base;
  using Base::init_;
  using Base::reset;
  using Base::predictionTimeline_;
  using Base::safe_;
  using Base::front_;
  using Base::readFromInfo;
  using Base::boolRegister_;
  using Base::intRegister_;
  using Base::doubleRegister_;
  using Base::mUpdates_;
  using Base::mPrediction_;
  using Base::stringRegister_;
  using Base::subHandlers_;
  using Base::updateToUpdateMeasOnly_;
  typedef typename Base::mtFilterState mtFilterState;
  typedef typename Base::mtPrediction mtPrediction;
  typedef typename Base::mtState mtState;
  rovio::MultiCamera<mtState::nCam_> multiCamera_;
  std::string cameraCalibrationFile_[mtState::nCam_];
  int depthTypeInt_;

  /*
   * Bsp: extension
   */
  typedef BSP_FeatureParams<Base> mt_BspFeatureParams;
  BSP_FeatureParams<Base> bsp_featureParams_[FILTERSTATE::mtState::nMax_];	/**<Bsp: array of feature parameters.*/
    static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,triplet_);	/**<Bsp: alias static properties.*/
    static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,sixlet_);	/**<Bsp: alias static properties.*/
  static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,pos_idx_);	/**<Bsp: alias static properties.*/
  static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,vel_idx_);	/**<Bsp: alias static properties.*/
  static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,acb_idx_);	/**<Bsp: alias static properties.*/
  static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,gyb_idx_);	/**<Bsp: alias static properties.*/
  static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,att_idx_);	/**<Bsp: alias static properties.*/
    static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,yaw_idx_); 	/**<Bsp: alias static properties.*/
  static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,vep_0_idx_);	/**<Bsp: alias static properties.*/
  static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,vea_0_idx_);	/**<Bsp: alias static properties.*/
  static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,fea_0_idx_);	/**<Bsp: alias static properties.*/
    static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,dep_0_idx_); /**<Bsp: alias static properties.*/
  static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,covMat_size_); /**<Bsp: alias static properties.*/
    static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,map_featureVisibilityThreshold_);	/**<Bsp: alias static properties.*/
    static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,cam_FoVx_);	/**<Bsp: alias static properties.*/
    static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,cam_FoVy_);	/**<Bsp: alias static properties.*/
    static constexpr SYMBOL_ALIAS_INIT(mt_BspFeatureParams,cam_FoVz_);	/**<Bsp: alias static properties.*/
  static constexpr bool bspFilter_ = BSPFILTER;	/**<Bsp: enabled bsp functionalities flag.*/
  octomap::OcTree* octree_ = nullptr;	/**<Bsp: belief map (octomap).*/
  double bsp_Ts_ = 0.05;		/**<Bsp: propagation sampling time.*/
  double xyz_damp_ = 0.1;
  double xy_gyr_max_ = 2*M_PI/3;
  double z_gyr_max_ = 2*M_PI/3;
  double xy_gyr_P_ = 10;
  double z_gyr_P_ = 5;
  double rollpitch_max_ = 20*M_PI/180;
  double xy_P_ = 25*0.017453293;
  double xy_D_ = 12.5*0.017453293;
  double z_P_ = 0.5*mPrediction_.g_[2];
  double z_D_ = 2.5;
  
  /** \brief Constructor. Initializes the filter.
   */
  RovioFilter(){
    updateToUpdateMeasOnly_ = true;
    std::get<0>(mUpdates_).setCamera(&multiCamera_);
    init_.setCamera(&multiCamera_);
    depthTypeInt_ = 1;
    subHandlers_.erase("Update0");
    subHandlers_["ImgUpdate"] = &std::get<0>(mUpdates_);
    subHandlers_.erase("Update1");
    subHandlers_["PoseUpdate"] = &std::get<1>(mUpdates_);
    boolRegister_.registerScalar("Common.doVECalibration",init_.state_.aux().doVECalibration_);
    intRegister_.registerScalar("Common.depthType",depthTypeInt_);

    /*
     * Bsp: get values (some alias BSP_FeatureParams static parameters)
     */
    doubleRegister_.registerScalar("Bsp.cam_FoVx",cam_FoVx_);
    doubleRegister_.registerScalar("Bsp.cam_FoVy",cam_FoVy_);
    doubleRegister_.registerScalar("Bsp.cam_FoVz",cam_FoVz_);
    intRegister_.registerScalar("Bsp.map_featureVisibilityThreshold",map_featureVisibilityThreshold_);
    doubleRegister_.registerScalar("Bsp.bsp_Ts",bsp_Ts_);
    doubleRegister_.registerScalar("Bsp.xyz_damp",xyz_damp_);
    doubleRegister_.registerScalar("Bsp.xy_gyr_max",xy_gyr_max_);
    doubleRegister_.registerScalar("Bsp.z_gyr_max",z_gyr_max_);
    doubleRegister_.registerScalar("Bsp.xy_gyr_P",xy_gyr_P_);
    doubleRegister_.registerScalar("Bsp.z_gyr_P",z_gyr_P_);
    doubleRegister_.registerScalar("Bsp.rollpitch_max",rollpitch_max_);
    doubleRegister_.registerScalar("Bsp.xy_P",xy_P_);
    doubleRegister_.registerScalar("Bsp.xy_D",xy_D_);
    doubleRegister_.registerScalar("Bsp.z_P",z_P_);
    doubleRegister_.registerScalar("Bsp.z_D",z_D_);

    for(int camID=0;camID<mtState::nCam_;camID++){
      cameraCalibrationFile_[camID] = "";
      stringRegister_.registerScalar("Camera" + std::to_string(camID) + ".CalibrationFile",cameraCalibrationFile_[camID]);
      doubleRegister_.registerVector("Camera" + std::to_string(camID) + ".MrMC",init_.state_.aux().MrMC_[camID]);
      doubleRegister_.registerQuaternion("Camera" + std::to_string(camID) + ".qCM",init_.state_.aux().qCM_[camID]);
      doubleRegister_.removeScalarByVar(init_.state_.MrMC(camID)(0));
      doubleRegister_.removeScalarByVar(init_.state_.MrMC(camID)(1));
      doubleRegister_.removeScalarByVar(init_.state_.MrMC(camID)(2));
      doubleRegister_.removeScalarByVar(init_.state_.qCM(camID).toImplementation().w());
      doubleRegister_.removeScalarByVar(init_.state_.qCM(camID).toImplementation().x());
      doubleRegister_.removeScalarByVar(init_.state_.qCM(camID).toImplementation().y());
      doubleRegister_.removeScalarByVar(init_.state_.qCM(camID).toImplementation().z());
      for(int j=0;j<3;j++){
        doubleRegister_.removeScalarByVar(init_.cov_(mtState::template getId<mtState::_vep>(camID)+j,mtState::template getId<mtState::_vep>(camID)+j));
        doubleRegister_.removeScalarByVar(init_.cov_(mtState::template getId<mtState::_vea>(camID)+j,mtState::template getId<mtState::_vea>(camID)+j));
        doubleRegister_.registerScalar("Init.Covariance.vep",init_.cov_(mtState::template getId<mtState::_vep>(camID)+j,mtState::template getId<mtState::_vep>(camID)+j));
        doubleRegister_.registerScalar("Init.Covariance.vea",init_.cov_(mtState::template getId<mtState::_vea>(camID)+j,mtState::template getId<mtState::_vea>(camID)+j));
      }
      doubleRegister_.registerVector("Camera" + std::to_string(camID) + ".MrMC",init_.state_.MrMC(camID));
      doubleRegister_.registerQuaternion("Camera" + std::to_string(camID) + ".qCM",init_.state_.qCM(camID));
    }
    for(int i=0;i<mtState::nPose_;i++){
      doubleRegister_.removeScalarByVar(init_.state_.poseLin(i)(0));
      doubleRegister_.removeScalarByVar(init_.state_.poseLin(i)(1));
      doubleRegister_.removeScalarByVar(init_.state_.poseLin(i)(2));
      doubleRegister_.removeScalarByVar(init_.state_.poseRot(i).toImplementation().w());
      doubleRegister_.removeScalarByVar(init_.state_.poseRot(i).toImplementation().x());
      doubleRegister_.removeScalarByVar(init_.state_.poseRot(i).toImplementation().y());
      doubleRegister_.removeScalarByVar(init_.state_.poseRot(i).toImplementation().z());
      for(int j=0;j<3;j++){
        doubleRegister_.removeScalarByVar(init_.cov_(mtState::template getId<mtState::_pop>(i)+j,mtState::template getId<mtState::_pop>(i)+j));
        doubleRegister_.removeScalarByVar(init_.cov_(mtState::template getId<mtState::_poa>(i)+j,mtState::template getId<mtState::_poa>(i)+j));
      }
    }
    if(std::get<1>(mUpdates_).inertialPoseIndex_>=0){
      std::get<1>(mUpdates_).doubleRegister_.registerVector("IrIW",init_.state_.poseLin(std::get<1>(mUpdates_).inertialPoseIndex_));
      std::get<1>(mUpdates_).doubleRegister_.registerQuaternion("qWI",init_.state_.poseRot(std::get<1>(mUpdates_).inertialPoseIndex_));
      for(int j=0;j<3;j++){
        std::get<1>(mUpdates_).doubleRegister_.registerScalar("init_cov_IrIW",init_.cov_(mtState::template getId<mtState::_pop>(std::get<1>(mUpdates_).inertialPoseIndex_)+j,mtState::template getId<mtState::_pop>(std::get<1>(mUpdates_).inertialPoseIndex_)+j));
        std::get<1>(mUpdates_).doubleRegister_.registerScalar("init_cov_qWI",init_.cov_(mtState::template getId<mtState::_poa>(std::get<1>(mUpdates_).inertialPoseIndex_)+j,mtState::template getId<mtState::_poa>(std::get<1>(mUpdates_).inertialPoseIndex_)+j));
        std::get<1>(mUpdates_).doubleRegister_.registerScalar("pre_cov_IrIW",mPrediction_.prenoiP_(mtPrediction::mtNoise::template getId<mtPrediction::mtNoise::_pop>(std::get<1>(mUpdates_).inertialPoseIndex_)+j,mtPrediction::mtNoise::template getId<mtPrediction::mtNoise::_pop>(std::get<1>(mUpdates_).inertialPoseIndex_)+j));
        std::get<1>(mUpdates_).doubleRegister_.registerScalar("pre_cov_qWI",mPrediction_.prenoiP_(mtPrediction::mtNoise::template getId<mtPrediction::mtNoise::_poa>(std::get<1>(mUpdates_).inertialPoseIndex_)+j,mtPrediction::mtNoise::template getId<mtPrediction::mtNoise::_poa>(std::get<1>(mUpdates_).inertialPoseIndex_)+j));
      }
    }
    if(std::get<1>(mUpdates_).bodyPoseIndex_>=0){
      std::get<1>(mUpdates_).doubleRegister_.registerVector("MrMV",init_.state_.poseLin(std::get<1>(mUpdates_).bodyPoseIndex_));
      std::get<1>(mUpdates_).doubleRegister_.registerQuaternion("qVM",init_.state_.poseRot(std::get<1>(mUpdates_).bodyPoseIndex_));
      for(int j=0;j<3;j++){
        std::get<1>(mUpdates_).doubleRegister_.registerScalar("init_cov_MrMV",init_.cov_(mtState::template getId<mtState::_pop>(std::get<1>(mUpdates_).bodyPoseIndex_)+j,mtState::template getId<mtState::_pop>(std::get<1>(mUpdates_).bodyPoseIndex_)+j));
        std::get<1>(mUpdates_).doubleRegister_.registerScalar("init_cov_qVM",init_.cov_(mtState::template getId<mtState::_poa>(std::get<1>(mUpdates_).bodyPoseIndex_)+j,mtState::template getId<mtState::_poa>(std::get<1>(mUpdates_).bodyPoseIndex_)+j));
        std::get<1>(mUpdates_).doubleRegister_.registerScalar("pre_cov_MrMV",mPrediction_.prenoiP_(mtPrediction::mtNoise::template getId<mtPrediction::mtNoise::_pop>(std::get<1>(mUpdates_).bodyPoseIndex_)+j,mtPrediction::mtNoise::template getId<mtPrediction::mtNoise::_pop>(std::get<1>(mUpdates_).bodyPoseIndex_)+j));
        std::get<1>(mUpdates_).doubleRegister_.registerScalar("pre_cov_qVM",mPrediction_.prenoiP_(mtPrediction::mtNoise::template getId<mtPrediction::mtNoise::_poa>(std::get<1>(mUpdates_).bodyPoseIndex_)+j,mtPrediction::mtNoise::template getId<mtPrediction::mtNoise::_poa>(std::get<1>(mUpdates_).bodyPoseIndex_)+j));
      }
    }
    int ind;
    for(int i=0;i<FILTERSTATE::mtState::nMax_;i++){
      ind = mtState::template getId<mtState::_fea>(i);
      doubleRegister_.removeScalarByVar(init_.cov_(ind,ind));
      doubleRegister_.removeScalarByVar(init_.cov_(ind+1,ind+1));
      ind = mtState::template getId<mtState::_fea>(i)+2;
      doubleRegister_.removeScalarByVar(init_.cov_(ind,ind));
      doubleRegister_.removeScalarByVar(init_.state_.dep(i).p_);
      doubleRegister_.removeScalarByVar(init_.state_.CfP(i).nor_.q_.toImplementation().w());
      doubleRegister_.removeScalarByVar(init_.state_.CfP(i).nor_.q_.toImplementation().x());
      doubleRegister_.removeScalarByVar(init_.state_.CfP(i).nor_.q_.toImplementation().y());
      doubleRegister_.removeScalarByVar(init_.state_.CfP(i).nor_.q_.toImplementation().z());
      std::get<0>(mUpdates_).intRegister_.registerScalar("statLocalQualityRange",init_.fsm_.features_[i].mpStatistics_->localQualityRange_);
      std::get<0>(mUpdates_).intRegister_.registerScalar("statLocalVisibilityRange",init_.fsm_.features_[i].mpStatistics_->localVisibilityRange_);
      std::get<0>(mUpdates_).intRegister_.registerScalar("statMinGlobalQualityRange",init_.fsm_.features_[i].mpStatistics_->minGlobalQualityRange_);
      std::get<0>(mUpdates_).boolRegister_.registerScalar("doPatchWarping",init_.state_.CfP(i).trackWarping_);
    }
    std::get<0>(mUpdates_).doubleRegister_.removeScalarByVar(std::get<0>(mUpdates_).outlierDetection_.getMahalTh(0));
    std::get<0>(mUpdates_).doubleRegister_.registerScalar("MahalanobisTh",std::get<0>(mUpdates_).outlierDetection_.getMahalTh(0));
    std::get<0>(mUpdates_).outlierDetection_.setEnabledAll(true);
    std::get<1>(mUpdates_).outlierDetection_.setEnabledAll(true);
    boolRegister_.registerScalar("Common.verbose",std::get<0>(mUpdates_).verbose_);
    mPrediction_.doubleRegister_.removeScalarByStr("alpha");
    mPrediction_.doubleRegister_.removeScalarByStr("beta");
    mPrediction_.doubleRegister_.removeScalarByStr("kappa");
    boolRegister_.registerScalar("PoseUpdate.doVisualization",init_.plotPoseMeas_);
    reset(0.0);
  }

  /** \brief Reloads the camera calibration for all cameras and resets the depth map type.
   */
  void refreshProperties(){
    if(std::get<0>(mUpdates_).useDirectMethod_){
      init_.mode_ = LWF::ModeIEKF;
    } else {
      init_.mode_ = LWF::ModeEKF;
    }
    for(int camID = 0;camID<mtState::nCam_;camID++){
      if (!cameraCalibrationFile_[camID].empty()) {
        multiCamera_.cameras_[camID].load(cameraCalibrationFile_[camID]);
      }
    }
    for(int i=0;i<FILTERSTATE::mtState::nMax_;i++){
      init_.state_.dep(i).setType(depthTypeInt_);
    }
    
    /*
     *Bsp: refresh static properties
     */
    mt_BspFeatureParams::refreshProperties();
  };

  /** \brief Destructor
   */
  virtual ~RovioFilter(){};
//  void resetToImuPose(V3D WrWM, QPD qMW, double t = 0.0){
//    init_.state_.initWithImuPose(WrWM,qMW);
//    reset(t);
//  }

  /** \brief Resets the filter with an accelerometer measurement.
   *
   *  @param fMeasInit - Accelerometer measurement.
   *  @param t         - Current time.
   */
  void resetWithAccelerometer(const V3D& fMeasInit, double t = 0.0){
    init_.initWithAccelerometer(fMeasInit);
    reset(t);
  }

  /** \brief Resets the filter with an external pose.
   *
   *  @param WrWM - Position Vector, pointing from the World-Frame to the IMU-Frame, expressed in World-Coordinates.
   *  @param qMW  - Quaternion, expressing World-Frame in IMU-Coordinates (World Coordinates->IMU Coordinates)
   *  @param t    - Current time.
   */
  void resetWithPose(V3D WrWM, QPD qMW, double t = 0.0) {
    init_.initWithImuPose(WrWM, qMW);
    reset(t);
  }

  /** \brief Sets the transformation between IMU and Camera.
   *
   *  @param R_VM  -  Rotation matrix, expressing the orientation of the IMU  in Camera Cooridinates (IMU Coordinates -> Camera Coordinates).
   *  @param VrVM  -  Vector, pointing from the camera frame to the IMU frame, expressed in IMU Coordinates.
   *  @param camID -  ID of the considered camera.
   */
  void setExtrinsics(const Eigen::Matrix3d& R_CM, const Eigen::Vector3d& CrCM, const int camID = 0){
    rot::RotationMatrixAD R(R_CM);
    init_.state_.aux().qCM_[camID] = QPD(R.getPassive());
    init_.state_.aux().MrMC_[camID] = -init_.state_.aux().qCM_[camID].inverseRotate(CrCM);
  }

  /** \brief Calculates D-optimality metric.
   *
   *  @param covMat  -  Covariance matrix.
   */
  double calcDopt(const Eigen::MatrixXd& covMat)
  {
    return std::exp(std::log(std::pow( covMat.determinant(), ( 1/static_cast<double>(covMat.rows()) )))); 
  }
  
  /** \brief Calculates covariance submatrix for D-optimality computation.
   *
   *  @param covMatIn   -  Original covariance matrix.
   *  @param covSubMat  -  Output covariance sub-matrix.
   *  @param octree     -  Map for visibility checking (only during propagation).
   */
  void calcCovSubMat(const Eigen::MatrixXd& covMatIn, Eigen::MatrixXd& covSubMat){

    static constexpr bool stateFlags_[] = {true, false, false, false, false, false, false, false, false}; //pos, vel, acb, gyb, att, vep, vea, fea
    static constexpr bool yawOnly_ = true;
    static_assert(yawOnly_ && (yawOnly_ != stateFlags_[mtState::_att]), "calcCovSubMat: yawOnly_ and useFlags[mtState::_att] both true.");
    static constexpr bool depthOnly_ = true;
    static_assert(depthOnly_ && (depthOnly_ != stateFlags_[mtState::_fea]), "calcCovSubMat: depthOnly_ and useFlags[mtState::_fea] both true.");
    //EIGEN_DEFAULT_DENSE_INDEX_TYPE is std::ptrdiff_t, TODO: get rid of MatrixXd?
    static constexpr Eigen::MatrixXd::Index subMat_size_ = 3*( static_cast<size_t>(stateFlags_[mtState::_pos]) + 
					                       static_cast<size_t>(stateFlags_[mtState::_vel]) + 
					                       static_cast<size_t>(stateFlags_[mtState::_acb]) + 
					                       static_cast<size_t>(stateFlags_[mtState::_gyb]) + 
					                       static_cast<size_t>(stateFlags_[mtState::_att]) ) + 
					                         static_cast<size_t>(yawOnly_ && !stateFlags_[mtState::_att]) + //yawOnly (special) case (exclusive via static_assert): if attitude was already enabled, no need to increase to accomodate yaw
					                   3*( static_cast<size_t>(stateFlags_[mtState::_vep])*mtState::nCam_ + 
					                       static_cast<size_t>(stateFlags_[mtState::_vea])*mtState::nCam_ ) + 
					                   3*( static_cast<size_t>(stateFlags_[mtState::_fea])*mtState::nMax_ ) + 
							         static_cast<size_t>(depthOnly_ && !stateFlags_[mtState::_fea])*mtState::nMax_ ; //depthOnly (special) case (exclusive via static_assert): if features are already enabled, no need to increase to accomodate depth
    
    static constexpr bool diagOnly_ = true;					
 
    const mtFilterState& filterState = safe_;
    const mtState& state = safe_.state_;

    Eigen::MatrixXd covMat_;
    if(diagOnly_){
      Eigen::VectorXd diagVec = covMatIn.diagonal();  //TODO: needed?
      covMat_ = Eigen::MatrixXd::Zero(covMat_size_,covMat_size_);
      covMat_ = diagVec.asDiagonal(); 
    }
    else{
      covMat_ = covMatIn;
    }

    covSubMat.resize(subMat_size_,subMat_size_);  //TODO: needed?
    covSubMat=Eigen::MatrixXd::Zero(subMat_size_,subMat_size_);

    Eigen::MatrixXd::Index sub_idx=0;
    if(stateFlags_[mtState::_pos]){
      covSubMat.block(sub_idx,sub_idx,triplet_,triplet_) = covMat_.block(pos_idx_,pos_idx_,triplet_,triplet_);
      sub_idx+=triplet_;
    }
    if(stateFlags_[mtState::_vel]){
      covSubMat.block(sub_idx,sub_idx,triplet_,triplet_) = covMat_.block(vel_idx_,vel_idx_,triplet_,triplet_);
      sub_idx+=triplet_;
    }
    if(stateFlags_[mtState::_acb]){
      covSubMat.block(sub_idx,sub_idx,triplet_,triplet_) = covMat_.block(acb_idx_,acb_idx_,triplet_,triplet_);
      sub_idx+=triplet_;
    }
    if(stateFlags_[mtState::_gyb]){
      covSubMat.block(sub_idx,sub_idx,triplet_,triplet_) = covMat_.block(gyb_idx_,gyb_idx_,triplet_,triplet_);
      sub_idx+=triplet_;
    }
    if (yawOnly_){ 
      covSubMat(sub_idx,sub_idx) = covMat_(yaw_idx_,yaw_idx_);
      ++sub_idx;
    }
    else if(stateFlags_[mtState::_att]){
      covSubMat.block(sub_idx,sub_idx,triplet_,triplet_) = covMat_.block(att_idx_,att_idx_,triplet_,triplet_);
      sub_idx+=triplet_;
    }
    if(stateFlags_[mtState::_vep] && stateFlags_[mtState::_vea]){ 
      covSubMat.block(sub_idx,sub_idx,sixlet_*mtState::nCam_,sixlet_*mtState::nCam_) = covMat_.block(vep_0_idx_,vep_0_idx_,sixlet_*mtState::nCam_,sixlet_*mtState::nCam_);
      sub_idx+=sixlet_*mtState::nCam_;
    }
    else if(stateFlags_[mtState::_vep]){
      for (unsigned int i=0; i<mtState::nCam_; ++i){
        Eigen::MatrixXd::Index covSubMatIdx = sub_idx + triplet_* i;
        Eigen::MatrixXd::Index covMatIdx = vep_0_idx_ + sixlet_ * i;
        covSubMat.block(covSubMatIdx,covSubMatIdx,triplet_,triplet_) = covMat_.block(covMatIdx,covMatIdx,triplet_,triplet_);
      }
      sub_idx+=triplet_*mtState::nCam_;
    }
    else if(stateFlags_[mtState::_vea]){
      for (unsigned int i=0; i<mtState::nCam_; ++i){
        Eigen::MatrixXd::Index covSubMatIdx = sub_idx + triplet_* i;
        Eigen::MatrixXd::Index covMatIdx = vea_0_idx_ + sixlet_ * i;
        covSubMat.block(covSubMatIdx,covSubMatIdx,triplet_,triplet_) = covMat_.block(covMatIdx,covMatIdx,triplet_,triplet_);
      }
      sub_idx+=triplet_*mtState::nCam_;
    }

    for (unsigned int i=0; i<mtState::nMax_; ++i){
      if( filterState.fsm_.isValid_[i] &&
          std::get<mt_BspFeatureParams::_bsp_fov>(bsp_featureParams_[i]) &&
          (octree_ == nullptr || std::get<mt_BspFeatureParams::_bsp_los>(bsp_featureParams_[i]) == bsp::LoS::Free) ){
        if(depthOnly_){ 
	  const double& feaCov = covMatIn(dep_0_idx_+triplet_*i,dep_0_idx_+triplet_*i);
	  if (feaCov){
	    covSubMat(sub_idx,sub_idx) = feaCov;
	    ++sub_idx;
	  }
        }
	else if(stateFlags_[mtState::_fea]){
	  const Eigen::Matrix3d& feaCov = covMatIn.block(fea_0_idx_+triplet_*i,fea_0_idx_+triplet_*i,triplet_,triplet_);
	  if (calcDopt(feaCov)){
	    covSubMat.block(sub_idx,sub_idx,triplet_,triplet_) = feaCov;
	    sub_idx += triplet_;
	  }
	}
      }
    }

    covSubMat.conservativeResize(sub_idx,sub_idx);
  }

};

}


#endif /* ROVIO_ROVIO_FILTER_HPP_ */
