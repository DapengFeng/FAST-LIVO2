/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_H_
#define VIO_H_

#include "voxel_map.h"
#include "feature.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/video/tracking.hpp>
#include <pcl/filters/voxel_grid.h>
#include <deque>
#include <set>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>

struct SubSparseMap
{
  vector<float> propa_errors;
  vector<float> errors;
  vector<vector<float>> warp_patch;
  vector<int> search_levels;
  vector<VisualPoint *> voxel_points;
  vector<double> inv_expo_list;
  vector<pointWithVar> add_from_voxel_map;

  SubSparseMap()
  {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
  };

  void reset()
  {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
  }
};

class Warp
{
public:
  Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}
  ~Warp() {}
};

class VOXEL_POINTS
{
public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS() 
  { 
    for (VisualPoint* vp : voxel_points) 
    {
      if (vp != nullptr) { delete vp; vp = nullptr; }
    }
  }
};

struct FeatureVioLandmark
{
  cv::Point2f px;
  V3D point_w;
  double depth = 0.0;
  double depth_px_dist = 0.0;
};

class VIOManager
{
public:
  int grid_size;
  vk::AbstractCamera *cam;
  vk::PinholeCamera *pinhole_cam;
  StatesGroup *state;
  StatesGroup *state_propagat;
  M3D Rli, Rci, Rcl, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
  V3D Pli, Pci, Pcl, Pcw;
  vector<int> grid_num;
  vector<int> map_index;
  vector<int> border_flag;
  vector<int> update_flag;
  vector<float> map_dist;
  vector<float> scan_value;
  vector<float> patch_buffer;
  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en, has_ref_patch_cache;
  bool ncc_en = false, colmap_output_en = false;
  bool saif_gate_en = false;
  bool photometric_update_en = true;
  bool feature_vio_diagnostic_en = false;
  bool lio_info_valid = false;

  int width, height, grid_n_width, grid_n_height, length;
  double image_resize_factor;
  double fx, fy, cx, cy;
  int patch_pyrimid_level, patch_size, patch_size_total, patch_size_half, border, warp_len;
  int max_iterations, total_points;

  double img_point_cov, outlier_threshold, ncc_thre;
  double saif_min_sqrt_info = 1.0;
  double saif_min_weight = 0.0;
  int lio_info_weak_dir = -1;
  Eigen::Matrix<double, 6, 1> lio_info_weak_vec = Eigen::Matrix<double, 6, 1>::Zero();
  int feature_vio_max_features = 500;
  int feature_vio_depth_search_radius = 4;
  double feature_vio_quality_level = 0.01;
  double feature_vio_min_distance = 12.0;
  double feature_vio_inlier_thresh_px = 3.0;
  double feature_vio_min_depth = 0.5;
  double feature_vio_max_depth = 120.0;
  bool feature_vio_fb_check_en = true;
  bool feature_vio_ransac_en = true;
  int feature_vio_lk_window_size = 21;
  int feature_vio_lk_max_level = 3;
  bool feature_vio_lk_pyramid_cache_en = true;
  bool feature_vio_lk_temporal_cache_en = true;
  bool feature_vio_depth_image_reuse_en = true;
  bool feature_vio_dry_run_en = true;
  bool feature_vio_update_en = false;
  bool feature_vio_inekf_update_en = false;
  int feature_vio_frame_stride = 1;
  double feature_vio_fb_thresh_px = 1.5;
  double feature_vio_ransac_thresh_px = 1.5;
  double feature_vio_gate_min_inlier_ratio = 0.35;
  double feature_vio_dry_run_damping = 1e-6;
  double feature_vio_dry_run_max_condition = 1e8;
  double feature_vio_img_point_cov = 100.0;
  bool feature_vio_adaptive_weight_en = false;
  double feature_vio_adaptive_huber_px = 1.5;
  double feature_vio_adaptive_depth_sigma_px = 2.0;
  double feature_vio_adaptive_min_weight = 0.05;
  double feature_vio_adaptive_depth_gate_px = 2.0;
  double feature_vio_adaptive_depth_scene_gate_px = 1.7;
  double feature_vio_update_scale = 1.0;
  double feature_vio_max_update_norm = 0.02;
  double feature_vio_max_rot_deg = 0.2;
  double feature_vio_max_pos_norm = 0.02;
  bool feature_vio_bias_watchdog_en = false;
  int feature_vio_bias_watchdog_window = 200;
  int feature_vio_bias_watchdog_min_samples = 30;
  double feature_vio_bias_watchdog_max_rot_mean = 1e-4;
  double feature_vio_bias_watchdog_max_pos_mean = 1e-4;
  int feature_vio_gate_min_inliers = 30;
  
  SubSparseMap *visual_submap;
  std::vector<std::vector<V3D>> rays_with_sample_points;

  double compute_jacobian_time, update_ekf_time;
  double ave_total = 0;
  size_t saif_gated_dirs = 0;
  double saif_frame_min_weight = 1.0;
  double saif_frame_weight_avg = 1.0;
  double saif_frame_weight_sum = 0.0;
  size_t saif_frame_samples = 0;
  size_t vio_map_pg_size = 0;
  size_t vio_map_normal_zero = 0;
  size_t vio_map_in_frame = 0;
  size_t vio_map_grid_map_skip = 0;
  size_t vio_map_selected_from_scan = 0;
  size_t vio_map_voxel_candidates = 0;
  size_t vio_map_selected_from_voxel = 0;
  size_t vio_map_added = 0;
  size_t vio_map_depth_positive = 0;
  double vio_map_shi_max = 0.0;
  double vio_map_u_min = 0.0;
  double vio_map_u_max = 0.0;
  double vio_map_v_min = 0.0;
  double vio_map_v_max = 0.0;
  double vio_map_z_min = 0.0;
  double vio_map_z_max = 0.0;
  size_t feature_vio_prev_landmarks = 0;
  size_t feature_vio_tracked = 0;
  size_t feature_vio_valid_reproj = 0;
  size_t feature_vio_inliers = 0;
  size_t feature_vio_new_features = 0;
  size_t feature_vio_depth_landmarks = 0;
  size_t feature_vio_fb_inliers = 0;
  size_t feature_vio_fb_rejects = 0;
  bool feature_vio_fb_check_run = false;
  size_t feature_vio_ransac_inliers = 0;
  size_t feature_vio_robust_inliers = 0;
  double feature_vio_reproj_rmse = 0.0;
  double feature_vio_inlier_ratio = 0.0;
  double feature_vio_robust_reproj_rmse = 0.0;
  double feature_vio_robust_inlier_ratio = 0.0;
  double feature_vio_robust_depth_mean = 0.0;
  double feature_vio_robust_depth_min = 0.0;
  double feature_vio_robust_depth_max = 0.0;
  double feature_vio_robust_ref_depth_mean = 0.0;
  double feature_vio_robust_depth_px_dist_mean = 0.0;
  double feature_vio_robust_reproj_bias_u = 0.0;
  double feature_vio_robust_reproj_bias_v = 0.0;
  double feature_vio_robust_abs_reproj_mean = 0.0;
  double feature_vio_mean_parallax = 0.0;
  size_t feature_vio_solve_points = 0;
  double feature_vio_dry_run_update_norm = 0.0;
  double feature_vio_dry_run_rot_deg = 0.0;
  double feature_vio_dry_run_pos_norm = 0.0;
  double feature_vio_dry_run_condition = 0.0;
  double feature_vio_dry_run_eig_min = 0.0;
  double feature_vio_dry_run_eig_max = 0.0;
  double feature_vio_weight_mean = 1.0;
  double feature_vio_weight_min = 1.0;
  double feature_vio_effective_points = 0.0;
  double feature_vio_adaptive_depth_scene_sum_ = 0.0;
  double feature_vio_adaptive_inlier_scene_sum_ = 0.0;
  size_t feature_vio_adaptive_depth_scene_samples_ = 0;
  double feature_vio_adaptive_depth_scene_mean = 0.0;
  double feature_vio_adaptive_inlier_scene_mean = 0.0;
  double feature_vio_commit_update_norm = 0.0;
  double feature_vio_commit_rot_deg = 0.0;
  double feature_vio_commit_pos_norm = 0.0;
  double feature_vio_commit_rot_x = 0.0;
  double feature_vio_commit_rot_y = 0.0;
  double feature_vio_commit_rot_z = 0.0;
  double feature_vio_commit_pos_x = 0.0;
  double feature_vio_commit_pos_y = 0.0;
  double feature_vio_commit_pos_z = 0.0;
  int feature_vio_lio_weak_dir = -1;
  double feature_vio_lio_weak_abs_projection = 0.0;
  double feature_vio_lio_weak_abs_cosine = 0.0;
  size_t feature_vio_bias_watchdog_samples = 0;
  double feature_vio_bias_watchdog_rot_mean_norm = 0.0;
  double feature_vio_bias_watchdog_pos_mean_norm = 0.0;
  bool feature_vio_bias_watchdog_reject = false;
  size_t feature_vio_saif_gated_dirs = 0;
  double feature_vio_saif_min_weight = 1.0;
  double feature_vio_saif_weight_avg = 1.0;
  double feature_vio_diag_time = 0.0;
  double feature_vio_track_time = 0.0;
  double feature_vio_solve_time = 0.0;
  double feature_vio_landmark_time = 0.0;
  double feature_vio_detect_time = 0.0;
  double feature_vio_depth_project_time = 0.0;
  double feature_vio_depth_match_time = 0.0;
  bool feature_vio_gate_pass = false;
  bool feature_vio_commit_pass = false;
  int feature_vio_dry_run_status = 0;
  int feature_vio_commit_status = 0;
  // double ave_build_residual_time = 0;
  // double ave_ekf_time = 0;

  int frame_count = 0;
  bool plot_flag;

  Eigen::Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
  MatrixXd K, H_sub_inv;

  ofstream fout_camera, fout_colmap;
  unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  unordered_map<VOXEL_LOCATION, int> sub_feat_map; 
  unordered_map<int, Warp *> warp_map;
  vector<VisualPoint *> retrieve_voxel_points;
  vector<pointWithVar> append_voxel_points;
  vector<FeatureVioLandmark> feature_vio_prev_landmarks_;
  std::deque<Eigen::Matrix<double, 6, 1>> feature_vio_bias_watchdog_window_;
  Eigen::Matrix<double, 6, 1> feature_vio_bias_watchdog_sum_ = Eigen::Matrix<double, 6, 1>::Zero();
  FramePtr new_frame_;
  cv::Mat img_cp, img_rgb, img_test, feature_vio_prev_img_, feature_vio_depth_img_, feature_vio_index_img_;
  std::vector<cv::Mat> feature_vio_prev_lk_pyramid_;
  int feature_vio_prev_lk_window_ = 0;
  int feature_vio_prev_lk_max_level_ = -1;

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN
  };

  VIOManager();
  ~VIOManager();
  void updateStateInverse(cv::Mat img, int level);
  void updateState(cv::Mat img, int level);
  void processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  void retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg);
  void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P);
  void initializeVIO();
  void getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);
  void computeProjectionJacobian(V3D p, MD(2, 3) & J);
  void computeJacobianAndUpdateEKF(cv::Mat img);
  void runFeatureVioDiagnostic(const cv::Mat &img, const vector<pointWithVar> &pg);
  void buildFeatureVioLandmarks(const cv::Mat &img, const vector<pointWithVar> &pg, vector<FeatureVioLandmark> &landmarks);
  void resetGrid();
  void updateVisualMapPoints(cv::Mat img);
  void getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref, const SE3 &T_cur_ref,
                           const int level_ref, 
                           const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref);
  void getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref,
                                     const V3D &xyz_ref, const V3D &normal_ref, const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref);
  void warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                  const int pyramid_level, const int halfpatch_size, float *patch);
  void insertPointIntoVoxelMap(VisualPoint *pt_new);
  void plotTrackedPoints();
  void updateFrameState(StatesGroup state);
  void projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void precomputeReferencePatches(int level);
  void dumpDataForColmap();
  double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);
  V3F getInterpolatedPixel(cv::Mat img, V2D pc);
  
  // void resetRvizDisplay();
  // deque<VisualPoint *> map_cur_frame;
  // deque<VisualPoint *> sub_map_ray;
  // deque<VisualPoint *> sub_map_ray_fov;
  // deque<VisualPoint *> visual_sub_map_cur;
  // deque<VisualPoint *> visual_converged_point;
  // std::vector<std::vector<V3D>> sample_points;

  // PointCloudXYZI::Ptr pg_down;
  // pcl::VoxelGrid<PointType> downSizeFilter;
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_H_
