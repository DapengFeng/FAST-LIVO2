/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio.h"

namespace
{

struct SaifGateStats
{
  size_t gated_dirs = 0;
  double min_weight = 1.0;
  double weight_sum = 0.0;
};

SaifGateStats applySaifGate(Eigen::Matrix<double, 6, 6> &info, Eigen::Matrix<double, 6, 1> &rhs, const double min_sqrt_info,
                            const double min_weight)
{
  SaifGateStats stats;
  stats.weight_sum = 6.0;
  if (min_sqrt_info <= 0.0) return stats;

  const Eigen::Matrix<double, 6, 6> sym_info = 0.5 * (info + info.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(sym_info);
  if (solver.info() != Eigen::Success) return stats;

  const Eigen::Matrix<double, 6, 1> eigs = solver.eigenvalues();
  const Eigen::Matrix<double, 6, 6> eigvecs = solver.eigenvectors();
  Eigen::Matrix<double, 6, 1> weights = Eigen::Matrix<double, 6, 1>::Ones();
  stats.weight_sum = 0.0;
  stats.min_weight = 1.0;
  for (int i = 0; i < 6; ++i)
  {
    const double sqrt_info = std::sqrt(std::max(eigs(i), 0.0));
    double weight = sqrt_info >= min_sqrt_info ? 1.0 : sqrt_info / min_sqrt_info;
    weight = std::max(min_weight, std::min(1.0, weight));
    weights(i) = weight;
    stats.weight_sum += weight;
    stats.min_weight = std::min(stats.min_weight, weight);
    if (weight < 1.0) stats.gated_dirs++;
  }

  const Eigen::Matrix<double, 6, 6> gate = eigvecs * weights.array().square().matrix().asDiagonal() * eigvecs.transpose();
  info = gate * sym_info;
  info = 0.5 * (info + info.transpose());
  rhs = gate * rhs;
  return stats;
}

} // namespace

VIOManager::VIOManager()
{
  // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
}

VIOManager::~VIOManager()
{
  delete visual_submap;
  for (auto& pair : warp_map) delete pair.second;
  warp_map.clear();
  for (auto& pair : feat_map) delete pair.second;
  feat_map.clear();
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot)
{
  Pli = -rot.transpose() * transl;
  Rli = rot.transpose();
}

void VIOManager::setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P)
{
  Rcl << MAT_FROM_ARRAY(R);
  Pcl << VEC_FROM_ARRAY(P);
}

void VIOManager::initializeVIO()
{
  visual_submap = new SubSparseMap;

  fx = cam->fx();
  fy = cam->fy();
  cx = cam->cx();
  cy = cam->cy();
  image_resize_factor = cam->scale();

  printf("intrinsic: %.6lf, %.6lf, %.6lf, %.6lf\n", fx, fy, cx, cy);

  width = cam->width();
  height = cam->height();

  printf("width: %d, height: %d, scale: %f\n", width, height, image_resize_factor);
  Rci = Rcl * Rli;
  Pci = Rcl * Pli + Pcl;

  V3D Pic;
  M3D tmp;
  Jdphi_dR = Rci;
  Pic = -Rci.transpose() * Pci;
  tmp << SKEW_SYM_MATRX(Pic);
  Jdp_dR = -Rci * tmp;

  if (grid_size > 10)
  {
    grid_n_width = ceil(static_cast<double>(width / grid_size));
    grid_n_height = ceil(static_cast<double>(height / grid_size));
  }
  else
  {
    grid_size = static_cast<int>(height / grid_n_height);
    grid_n_height = ceil(static_cast<double>(height / grid_size));
    grid_n_width = ceil(static_cast<double>(width / grid_size));
  }
  length = grid_n_width * grid_n_height;

  if(raycast_en)
  {
    // cv::Mat img_test = cv::Mat::zeros(height, width, CV_8UC1);
    // uchar* it = (uchar*)img_test.data;

    border_flag.resize(length, 0);

    std::vector<std::vector<V3D>>().swap(rays_with_sample_points);
    rays_with_sample_points.reserve(length);
    printf("grid_size: %d, grid_n_height: %d, grid_n_width: %d, length: %d\n", grid_size, grid_n_height, grid_n_width, length);

    float d_min = 0.1;
    float d_max = 3.0;
    float step = 0.2;
    for (int grid_row = 1; grid_row <= grid_n_height; grid_row++)
    {
      for (int grid_col = 1; grid_col <= grid_n_width; grid_col++)
      {
        std::vector<V3D> SamplePointsEachGrid;
        int index = (grid_row - 1) * grid_n_width + grid_col - 1;

        if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height || grid_col == grid_n_width) border_flag[index] = 1;

        int u = grid_size / 2 + (grid_col - 1) * grid_size;
        int v = grid_size / 2 + (grid_row - 1) * grid_size;
        // it[ u + v * width ] = 255;
        for (float d_temp = d_min; d_temp <= d_max; d_temp += step)
        {
          V3D xyz;
          xyz = cam->cam2world(u, v);
          xyz *= d_temp / xyz[2];
          // xyz[0] = (u - cx) / fx * d_temp;
          // xyz[1] = (v - cy) / fy * d_temp;
          // xyz[2] = d_temp;
          SamplePointsEachGrid.push_back(xyz);
        }
        rays_with_sample_points.push_back(SamplePointsEachGrid);
      }
    }
    // printf("rays_with_sample_points: %d, RaysWithSamplePointsCapacity: %d,
    // rays_with_sample_points[0].capacity(): %d, rays_with_sample_points[0]: %d\n",
    // rays_with_sample_points.size(), rays_with_sample_points.capacity(),
    // rays_with_sample_points[0].capacity(), rays_with_sample_points[0].size()); for
    // (const auto & it : rays_with_sample_points[0]) cout << it.transpose() << endl;
    // cv::imshow("img_test", img_test);
    // cv::waitKey(1);
  }

  if(colmap_output_en)
  {
    pinhole_cam = dynamic_cast<vk::PinholeCamera*>(cam);
    fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
    fout_colmap << "# Image list with two lines of data per image:\n";
    fout_colmap << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
    fout_camera << "# Camera list with one line of data per camera:\n";
    fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    fout_camera << "1 PINHOLE " << width << " " << height << " "
        << std::fixed << std::setprecision(6)  // 控制浮点数精度为10位
        << fx << " " << fy << " "
        << cx << " " << cy << std::endl;
    fout_camera.close();
  }
  grid_num.resize(length);
  map_index.resize(length);
  map_dist.resize(length);
  update_flag.resize(length);
  scan_value.resize(length);

  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  warp_len = patch_size_total * patch_pyrimid_level;
  border = (patch_size_half + 1) * (1 << patch_pyrimid_level);

  retrieve_voxel_points.reserve(length);
  append_voxel_points.reserve(length);

  sub_feat_map.clear();
}

void VIOManager::resetGrid()
{
  fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
  fill(map_index.begin(), map_index.end(), 0);
  fill(map_dist.begin(), map_dist.end(), 10000.0f);
  fill(update_flag.begin(), update_flag.end(), 0);
  fill(scan_value.begin(), scan_value.end(), 0.0f);

  retrieve_voxel_points.clear();
  retrieve_voxel_points.resize(length);

  append_voxel_points.clear();
  append_voxel_points.resize(length);

  total_points = 0;
}

// void VIOManager::resetRvizDisplay()
// {
  // sub_map_ray.clear();
  // sub_map_ray_fov.clear();
  // visual_sub_map_cur.clear();
  // visual_converged_point.clear();
  // map_cur_frame.clear();
  // sample_points.clear();
// }

void VIOManager::computeProjectionJacobian(V3D p, MD(2, 3) & J)
{
  const double x = p[0];
  const double y = p[1];
  const double z_inv = 1. / p[2];
  const double z_inv_2 = z_inv * z_inv;
  J(0, 0) = fx * z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -fx * x * z_inv_2;
  J(1, 0) = 0.0;
  J(1, 1) = fy * z_inv;
  J(1, 2) = -fy * y * z_inv_2;
}

void VIOManager::getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int scale = (1 << level);
  const int u_ref_i = floorf(pc[0] / scale) * scale;
  const int v_ref_i = floorf(pc[1] / scale) * scale;
  const float subpix_u_ref = (u_ref - u_ref_i) / scale;
  const float subpix_v_ref = (v_ref - v_ref_i) / scale;
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  for (int x = 0; x < patch_size; x++)
  {
    uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i - patch_size_half * scale + x * scale) * width + (u_ref_i - patch_size_half * scale);
    for (int y = 0; y < patch_size; y++, img_ptr += scale)
    {
      patch_tmp[patch_size_total * level + x * patch_size + y] =
          w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
    }
  }
}

void VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new)
{
  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
  double voxel_size = 0.5;
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = pt_w[j] / voxel_size;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  auto iter = feat_map.find(position);
  if (iter != feat_map.end())
  {
    iter->second->voxel_points.push_back(pt_new);
    iter->second->count++;
  }
  else
  {
    VOXEL_POINTS *ot = new VOXEL_POINTS(0);
    ot->voxel_points.push_back(pt_new);
    feat_map[position] = ot;
  }
}

void VIOManager::getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref, const V3D &xyz_ref, const V3D &normal_ref,
                                                  const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref)
{
  // create homography matrix
  const V3D t = T_cur_ref.inverse().translation();
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.rotationMatrix() * (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
  // Compute affine warp matrix A_ref_cur using homography projection
  const int kHalfPatchSize = 4;
  V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));
  V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));
  //   f_du_ref = f_du_ref/f_du_ref[2];
  //   f_dv_ref = f_dv_ref/f_dv_ref[2];
  const V3D f_cur(H_cur_ref * xyz_ref);
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  V2D px_cur(cam.world2cam(f_cur));
  V2D px_du_cur(cam.world2cam(f_du_cur));
  V2D px_dv_cur(cam.world2cam(f_dv_cur));
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref,
                                        const SE3 &T_cur_ref, const int level_ref, const int pyramid_level, const int halfpatch_size,
                                        Matrix2d &A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                               const int pyramid_level, const int halfpatch_size, float *patch)
{
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (isnan(A_ref_cur(0, 0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }

  float *patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y)
  {
    for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
    {
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << search_level);
      px_patch *= (1 << pyramid_level);
      const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
      else
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = (float)vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

int VIOManager::getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while (D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

double VIOManager::calculateNCC(float *ref_patch, float *cur_patch, int patch_size)
{
  double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
  double mean_ref = sum_ref / patch_size;

  double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
  double mean_curr = sum_cur / patch_size;

  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patch_size; i++)
  {
    double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
    numerator += n;
    demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
    demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

void VIOManager::retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (feat_map.size() <= 0) return;
  double ts0 = omp_get_wtime();

  // pg_down->reserve(feat_map.size());
  // downSizeFilter.setInputCloud(pg);
  // downSizeFilter.filter(*pg_down);

  // resetRvizDisplay();
  visual_submap->reset();

  // Controls whether to include the visual submap from the previous frame.
  sub_feat_map.clear();

  float voxel_size = 0.5;

  if (!normal_en) warp_map.clear();

  cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
  float *it = (float *)depth_img.data;

  // float it[height * width] = {0.0};

  // double t_insert, t_depth, t_position;
  // t_insert=t_depth=t_position=0;

  int loc_xyz[3];

  // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);
  // double ts1 = omp_get_wtime();

  // printf("pg size: %zu \n", pg.size());

  for (int i = 0; i < pg.size(); i++)
  {
    // double t0 = omp_get_wtime();

    V3D pt_w = pg[i].point_w;

    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    // t_position += omp_get_wtime()-t0;
    // double t1 = omp_get_wtime();

    auto iter = sub_feat_map.find(position);
    if (iter == sub_feat_map.end()) { sub_feat_map[position] = 0; }
    else { iter->second = 0; }

    // t_insert += omp_get_wtime()-t1;
    // double t2 = omp_get_wtime();

    V3D pt_c(new_frame_->w2f(pt_w));

    if (pt_c[2] > 0)
    {
      V2D px;
      // px[0] = fx * pt_c[0]/pt_c[2] + cx;
      // px[1] = fy * pt_c[1]/pt_c[2]+ cy;
      px = new_frame_->cam_->world2cam(pt_c);

      if (new_frame_->cam_->isInFrame(px.cast<int>(), border))
      {
        // cv::circle(img_cp, cv::Point2f(px[0], px[1]), 3, cv::Scalar(0, 0, 255), -1, 8);
        float depth = pt_c[2];
        int col = int(px[0]);
        int row = int(px[1]);
        it[width * row + col] = depth;
      }
    }
    // t_depth += omp_get_wtime()-t2;
  }

  // imshow("depth_img", depth_img);
  // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
  // printf("A11. calculate pt position: %.6lf \n", t_position);
  // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
  // printf("A13. generate depth map: %.6lf \n", t_depth);
  // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);

  // double t1 = omp_get_wtime();
  vector<VOXEL_LOCATION> DeleteKeyList;

  for (auto &iter : sub_feat_map)
  {
    VOXEL_LOCATION position = iter.first;

    // double t4 = omp_get_wtime();
    auto corre_voxel = feat_map.find(position);
    // double t5 = omp_get_wtime();

    if (corre_voxel != feat_map.end())
    {
      bool voxel_in_fov = false;
      std::vector<VisualPoint *> &voxel_points = corre_voxel->second->voxel_points;
      int voxel_num = voxel_points.size();

      for (int i = 0; i < voxel_num; i++)
      {
        VisualPoint *pt = voxel_points[i];
        if (pt == nullptr) continue;
        if (pt->obs_.size() == 0) continue;

        V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() * pt->normal_);
        V3D dir(new_frame_->T_f_w_ * pt->pos_);
        if (dir[2] < 0) continue;
        // dir.normalize();
        // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree  0.17 80 degree 0.08 85 degree

        V2D pc(new_frame_->w2c(pt->pos_));
        if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
        {
          // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 255, 255), -1, 8);
          voxel_in_fov = true;
          int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
          grid_num[index] = TYPE_MAP;
          Vector3d obs_vec(new_frame_->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          if (cur_dist <= map_dist[index])
          {
            map_dist[index] = cur_dist;
            retrieve_voxel_points[index] = pt;
          }
        }
      }
      if (!voxel_in_fov) { DeleteKeyList.push_back(position); }
    }
  }

  // RayCasting Module
  if (raycast_en)
  {
    for (int i = 0; i < length; i++)
    {
      if (grid_num[i] == TYPE_MAP || border_flag[i] == 1) continue;

      // int row = static_cast<int>(i / grid_n_width) * grid_size + grid_size /
      // 2; int col = (i - static_cast<int>(i / grid_n_width) * grid_n_width) *
      // grid_size + grid_size / 2;

      // cv::circle(img_cp, cv::Point2f(col, row), 3, cv::Scalar(255, 255, 0),
      // -1, 8);

      // vector<V3D> sample_points_temp;
      // bool add_sample = false;

      for (const auto &it : rays_with_sample_points[i])
      {
        V3D sample_point_w = new_frame_->f2w(it);
        // sample_points_temp.push_back(sample_point_w);

        for (int j = 0; j < 3; j++)
        {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto corre_sub_feat_map = sub_feat_map.find(sample_pos);
        if (corre_sub_feat_map != sub_feat_map.end()) break;

        auto corre_feat_map = feat_map.find(sample_pos);
        if (corre_feat_map != feat_map.end())
        {
          bool voxel_in_fov = false;

          std::vector<VisualPoint *> &voxel_points = corre_feat_map->second->voxel_points;
          int voxel_num = voxel_points.size();
          if (voxel_num == 0) continue;

          for (int j = 0; j < voxel_num; j++)
          {
            VisualPoint *pt = voxel_points[j];

            if (pt == nullptr) continue;
            if (pt->obs_.size() == 0) continue;

            // sub_map_ray.push_back(pt); // cloud_visual_sub_map
            // add_sample = true;

            V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() * pt->normal_);
            V3D dir(new_frame_->T_f_w_ * pt->pos_);
            if (dir[2] < 0) continue;
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17 80 degree 0.08 85 degree

            V2D pc(new_frame_->w2c(pt->pos_));

            if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
            {
              // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(255, 255, 0), -1, 8); 
              // sub_map_ray_fov.push_back(pt);

              voxel_in_fov = true;
              int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
              grid_num[index] = TYPE_MAP;
              Vector3d obs_vec(new_frame_->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              if (cur_dist <= map_dist[index])
              {
                map_dist[index] = cur_dist;
                retrieve_voxel_points[index] = pt;
              }
            }
          }

          if (voxel_in_fov) sub_feat_map[sample_pos] = 0;
          break;
        }
        else
        {
          VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
          auto iter = plane_map.find(sample_pos);
          if (iter != plane_map.end())
          {
            VoxelOctoTree *current_octo;
            current_octo = iter->second->find_correspond(sample_point_w);
            if (current_octo->plane_ptr_->is_plane_)
            {
              pointWithVar plane_center;
              VoxelPlane &plane = *current_octo->plane_ptr_;
              plane_center.point_w = plane.center_;
              plane_center.normal = plane.normal_;
              visual_submap->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
      // if(add_sample) sample_points.push_back(sample_points_temp);
    }
  }

  for (auto &key : DeleteKeyList)
  {
    sub_feat_map.erase(key);
  }

  // double t2 = omp_get_wtime();

  // cout<<"B. feat_map.find: "<<t2-t1<<endl;

  // double t_2, t_3, t_4, t_5;
  // t_2=t_3=t_4=t_5=0;

  for (int i = 0; i < length; i++)
  {
    if (grid_num[i] == TYPE_MAP)
    {
      // double t_1 = omp_get_wtime();

      VisualPoint *pt = retrieve_voxel_points[i];
      // visual_sub_map_cur.push_back(pt); // before

      V2D pc(new_frame_->w2c(pt->pos_));

      // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 0, 255), -1, 8); // Green Sparse Align tracked

      V3D pt_cam(new_frame_->w2f(pt->pos_));
      bool depth_continous = false;
      for (int u = -patch_size_half; u <= patch_size_half; u++)
      {
        for (int v = -patch_size_half; v <= patch_size_half; v++)
        {
          if (u == 0 && v == 0) continue;

          float depth = it[width * (v + int(pc[1])) + u + int(pc[0])];

          if (depth == 0.) continue;

          double delta_dist = abs(pt_cam[2] - depth);

          if (delta_dist > 0.5)
          {
            depth_continous = true;
            break;
          }
        }
        if (depth_continous) break;
      }
      if (depth_continous) continue;

      // t_2 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();
      Feature *ref_ftr;
      std::vector<float> patch_wrap(warp_len);

      int search_level;
      Matrix2d A_cur_ref_zero;

      if (!pt->is_normal_initialized_) continue;

      if (normal_en)
      {
        float phtometric_errors_min = std::numeric_limits<float>::max();

        if (pt->obs_.size() == 1)
        {
          ref_ftr = *pt->obs_.begin();
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else if (!pt->has_ref_patch_)
        {
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
          {
            Feature *ref_patch_temp = *it;
            float *patch_temp = ref_patch_temp->patch_;
            float phtometric_errors = 0.0;
            int count = 0;
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
            {
              if ((*itm)->id_ == ref_patch_temp->id_) continue;
              float *patch_cache = (*itm)->patch_;

              for (int ind = 0; ind < patch_size_total; ind++)
              {
                phtometric_errors += (patch_temp[ind] - patch_cache[ind]) * (patch_temp[ind] - patch_cache[ind]);
              }
              count++;
            }
            phtometric_errors = phtometric_errors / count;
            if (phtometric_errors < phtometric_errors_min)
            {
              phtometric_errors_min = phtometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else { ref_ftr = pt->ref_patch; }
      }
      else
      {
        if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;
      }

      if (normal_en)
      {
        V3D norm_vec = (ref_ftr->T_f_w_.rotationMatrix() * pt->normal_).normalized();
        
        V3D pf(ref_ftr->T_f_w_ * pt->pos_);
        // V3D pf_norm = pf.normalized();
        
        // double cos_theta = norm_vec.dot(pf_norm);
        // if(cos_theta < 0) norm_vec = -norm_vec;
        // if (abs(cos_theta) < 0.08) continue; // 0.5 60 degree 0.34 70 degree 0.17 80 degree 0.08 85 degree

        SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();

        getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);

        search_level = getBestSearchLevel(A_cur_ref_zero, 2);
      }
      else
      {
        auto iter_warp = warp_map.find(ref_ftr->id_);
        if (iter_warp != warp_map.end())
        {
          search_level = iter_warp->second->search_level;
          A_cur_ref_zero = iter_warp->second->A_cur_ref;
        }
        else
        {
          getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(),
                              ref_ftr->level_, 0, patch_size_half, A_cur_ref_zero);

          search_level = getBestSearchLevel(A_cur_ref_zero, 2);

          Warp *ot = new Warp(search_level, A_cur_ref_zero);
          warp_map[ref_ftr->id_] = ot;
        }
      }
      // t_4 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();

      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++)
      {
        warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_, search_level, pyramid_level, patch_size_half, patch_wrap.data());
      }

      getImagePatch(img, pc, patch_buffer.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]) *
                 (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]);
      }

      if (ncc_en)
      {
        double ncc = calculateNCC(patch_wrap.data(), patch_buffer.data(), patch_size_total);
        if (ncc < ncc_thre)
        {
          // grid_num[i] = TYPE_UNKNOWN;
          continue;
        }
      }

      if (error > outlier_threshold * patch_size_total) continue;

      visual_submap->voxel_points.push_back(pt);
      visual_submap->propa_errors.push_back(error);
      visual_submap->search_levels.push_back(search_level);
      visual_submap->errors.push_back(error);
      visual_submap->warp_patch.push_back(patch_wrap);
      visual_submap->inv_expo_list.push_back(ref_ftr->inv_expo_time_);

      // t_5 += omp_get_wtime() - t_1;
    }
  }
  total_points = visual_submap->voxel_points.size();

  // double t3 = omp_get_wtime();
  // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
  // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4
  // "<<t_5<<endl;
  printf("[ VIO ] Retrieve %d points from visual sparse map\n", total_points);
}

void VIOManager::computeJacobianAndUpdateEKF(cv::Mat img)
{
  compute_jacobian_time = update_ekf_time = 0.0;
  saif_gated_dirs = 0;
  saif_frame_min_weight = 1.0;
  saif_frame_weight_avg = 1.0;
  saif_frame_weight_sum = 0.0;
  saif_frame_samples = 0;
  if (total_points == 0) return;

  for (int level = patch_pyrimid_level - 1; level >= 0; level--)
  {
    if (inverse_composition_en)
    {
      has_ref_patch_cache = false;
      updateStateInverse(img, level);
    }
    else
      updateState(img, level);
  }
  if (saif_frame_samples > 0) saif_frame_weight_avg = saif_frame_weight_sum / static_cast<double>(saif_frame_samples);
  state->cov -= G * state->cov;
  updateFrameState(*state);
}

void VIOManager::generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg)
{
  vio_map_pg_size = pg.size();
  vio_map_normal_zero = 0;
  vio_map_in_frame = 0;
  vio_map_grid_map_skip = 0;
  vio_map_selected_from_scan = 0;
  vio_map_voxel_candidates = visual_submap->add_from_voxel_map.size();
  vio_map_selected_from_voxel = 0;
  vio_map_added = 0;
  vio_map_depth_positive = 0;
  vio_map_shi_max = 0.0;
  vio_map_u_min = std::numeric_limits<double>::max();
  vio_map_u_max = std::numeric_limits<double>::lowest();
  vio_map_v_min = std::numeric_limits<double>::max();
  vio_map_v_max = std::numeric_limits<double>::lowest();
  vio_map_z_min = std::numeric_limits<double>::max();
  vio_map_z_max = std::numeric_limits<double>::lowest();

  if (pg.size() <= 10) return;

  // double t0 = omp_get_wtime();
  for (int i = 0; i < pg.size(); i++)
  {
    if (pg[i].normal == V3D(0, 0, 0))
    {
      vio_map_normal_zero++;
      continue;
    }

    V3D pt = pg[i].point_w;
    const V3D pf = new_frame_->w2f(pt);
    vio_map_z_min = std::min(vio_map_z_min, pf[2]);
    vio_map_z_max = std::max(vio_map_z_max, pf[2]);
    if (pf[2] > 0.0) vio_map_depth_positive++;
    V2D pc(new_frame_->w2c(pt));
    vio_map_u_min = std::min(vio_map_u_min, pc[0]);
    vio_map_u_max = std::max(vio_map_u_max, pc[0]);
    vio_map_v_min = std::min(vio_map_v_min, pc[1]);
    vio_map_v_max = std::max(vio_map_v_max, pc[1]);

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      vio_map_in_frame++;
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        vio_map_shi_max = std::max(vio_map_shi_max, static_cast<double>(cur_value));
        // if (cur_value < 5) continue;
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = pg[i];
          grid_num[index] = TYPE_POINTCLOUD;
          vio_map_selected_from_scan++;
        }
      }
      else
      {
        vio_map_grid_map_skip++;
      }
    }
  }

  for (int j = 0; j < visual_submap->add_from_voxel_map.size(); j++)
  {
    V3D pt = visual_submap->add_from_voxel_map[j].point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        vio_map_shi_max = std::max(vio_map_shi_max, static_cast<double>(cur_value));
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = visual_submap->add_from_voxel_map[j];
          grid_num[index] = TYPE_POINTCLOUD;
          vio_map_selected_from_voxel++;
        }
      }
      else
      {
        vio_map_grid_map_skip++;
      }
    }
  }

  // double t_b1 = omp_get_wtime() - t0;
  // t0 = omp_get_wtime();

  int add = 0;
  for (int i = 0; i < length; i++)
  {
    if (grid_num[i] == TYPE_POINTCLOUD) // && (scan_value[i]>=50))
    {
      pointWithVar pt_var = append_voxel_points[i];
      V3D pt = pt_var.point_w;

      V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() * pt_var.normal);
      V3D dir(new_frame_->T_f_w_ * pt);
      dir.normalize();
      double cos_theta = dir.dot(norm_vec);
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      V2D pc(new_frame_->w2c(pt));

      float *patch = new float[patch_size_total];
      getImagePatch(img, pc, patch, 0);

      VisualPoint *pt_new = new VisualPoint(pt);

      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_, 0);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;

      pt_new->addFrameRef(ftr_new);
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      if (cos_theta < 0) { pt_new->normal_ = -pt_var.normal; }
      else { pt_new->normal_ = pt_var.normal; }
      
      pt_new->previous_normal_ = pt_new->normal_;

      insertPointIntoVoxelMap(pt_new);
      add += 1;
      // map_cur_frame.push_back(pt_new);
    }
  }
  vio_map_added = add;
  if (vio_map_depth_positive == 0)
  {
    vio_map_u_min = vio_map_u_max = vio_map_v_min = vio_map_v_max = vio_map_z_min = vio_map_z_max = 0.0;
  }

  // double t_b2 = omp_get_wtime() - t0;

  printf("[ VIO ] Append %d new visual map points\n", add);
  printf("[VIO_MAP_STATS] frame=%d pg_size=%zu normal_zero=%zu in_frame=%zu grid_map_skip=%zu selected_scan=%zu "
         "voxel_candidates=%zu selected_voxel=%zu added=%zu depth_positive=%zu shi_max=%.3f "
         "u_min=%.3f u_max=%.3f v_min=%.3f v_max=%.3f z_min=%.3f z_max=%.3f\n",
         frame_count, vio_map_pg_size, vio_map_normal_zero, vio_map_in_frame, vio_map_grid_map_skip, vio_map_selected_from_scan,
         vio_map_voxel_candidates, vio_map_selected_from_voxel, vio_map_added, vio_map_depth_positive, vio_map_shi_max, vio_map_u_min,
         vio_map_u_max, vio_map_v_min, vio_map_v_max, vio_map_z_min, vio_map_z_max);
  // printf("pg.size: %d \n", pg.size());
  // printf("B1. : %.6lf \n", t_b1);
  // printf("B2. : %.6lf \n", t_b2);
}

void VIOManager::updateVisualMapPoints(cv::Mat img)
{
  if (total_points == 0) return;

  int update_num = 0;
  SE3 pose_cur = new_frame_->T_f_w_;
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr) continue;
    if (pt->is_converged_)
    { 
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D pc(new_frame_->w2c(pt->pos_));
    bool add_flag = false;
    
    float *patch_temp = new float[patch_size_total];
    getImagePatch(img, pc, patch_temp, 0);
    // TODO: condition: distance and view_angle
    // Step 1: time
    Feature *last_feature = pt->obs_.back();
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    // Step 2: delta_pose
    SE3 pose_ref = last_feature->T_f_w_;
    SE3 delta_pose = pose_ref * pose_cur.inverse();
    double delta_p = delta_pose.translation().norm();
    double delta_theta = (delta_pose.rotationMatrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotationMatrix().trace() - 1));
    if (delta_p > 0.5 || delta_theta > 0.3) add_flag = true; // 0.5 || 0.3

    // Step 3: pixel distance
    Vector2d last_px = last_feature->px_;
    double pixel_dist = (pc - last_px).norm();
    if (pixel_dist > 40) add_flag = true;

    // Maintain the size of 3D point observation features.
    if (pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(new_frame_->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
      // cout<<"pt->obs_.size() exceed 20 !!!!!!"<<endl;
    }
    if (add_flag)
    {
      update_num += 1;
      update_flag[i] = 1;
      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, new_frame_->T_f_w_, visual_submap->search_levels[i]);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;
      pt->addFrameRef(ftr_new);
    }
  }
  printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

void VIOManager::updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;

  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue;
    if (update_flag[i] == 0) continue;

    const V3D &p_w = pt->pos_;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_w[j] / 0.5;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = plane_map.find(position);
    if (iter != plane_map.end())
    {
      VoxelOctoTree *current_octo;
      current_octo = iter->second->find_correspond(p_w);
      if (current_octo->plane_ptr_->is_plane_)
      {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        float dis_to_plane = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        float dis_to_plane_abs = fabs(dis_to_plane);
        float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
                              (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) + (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
        if (range_dis <= 3 * plane.radius_)
        {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l += plane.normal_.transpose() * pt->covariance_ * plane.normal_;

          if (dis_to_plane_abs < 3 * sqrt(sigma_l))
          {
            // V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() * plane.normal_);
            // V3D pf(new_frame_->T_f_w_ * pt->pos_);
            // V3D pf_ref(pt->ref_patch->T_f_w_ * pt->pos_);
            // V3D norm_vec_ref(pt->ref_patch->T_f_w_.rotationMatrix() *
            // plane.normal); double cos_ref = pf_ref.dot(norm_vec_ref);
            
            if (pt->previous_normal_.dot(plane.normal_) < 0) { pt->normal_ = -plane.normal_; }
            else { pt->normal_ = plane.normal_; }

            double normal_update = (pt->normal_ - pt->previous_normal_).norm();

            pt->previous_normal_ = pt->normal_;

            if (normal_update < 0.0001 && pt->obs_.size() > 10)
            {
              pt->is_converged_ = true;
              // visual_converged_point.push_back(pt);
            }
          }
        }
      }
    }

    float score_max = -1000.;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
    {
      Feature *ref_patch_temp = *it;
      float *patch_temp = ref_patch_temp->patch_;
      float NCC_up = 0.0;
      float NCC_down1 = 0.0;
      float NCC_down2 = 0.0;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotationMatrix() * pt->normal_;
      pf.normalize();
      double cos_angle = pf.dot(norm_vec);
      // if(fabs(cos_angle) < 0.86) continue; // 20 degree

      float ref_mean;
      if (abs(ref_patch_temp->mean_) < 1e-6)
      {
        float ref_sum = std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0);
        ref_mean = ref_sum / patch_size_total;
        ref_patch_temp->mean_ = ref_mean;
      }

      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
      {
        if ((*itm)->id_ == ref_patch_temp->id_) continue;
        float *patch_cache = (*itm)->patch_;

        float other_mean;
        if (abs((*itm)->mean_) < 1e-6)
        {
          float other_sum = std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0);
          other_mean = other_sum / patch_size_total;
          (*itm)->mean_ = other_mean;
        }

        for (int ind = 0; ind < patch_size_total; ind++)
        {
          NCC_up += (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 += (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 += (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2));
        count++;
      }

      NCC = NCC / count;

      score = NCC + cos_angle;

      ref_patch_temp->score_ = score;

      if (score > score_max)
      {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }

  }
}

void VIOManager::projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;
  // if(new_frame_->id_ != 2) return; //124

  int patch_size = 25;
  string dir = string(ROOT_DIR) + "Log/ref_cur_combine/";

  cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_normal = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_dense = cv::Mat::zeros(height, width, CV_8UC1);

  cv::Mat img_photometric_error = new_frame_->img_.clone();

  uchar *it = (uchar *)result.data;
  uchar *it_normal = (uchar *)result_normal.data;
  uchar *it_dense = (uchar *)result_dense.data;

  struct pixel_member
  {
    Vector2f pixel_pos;
    uint8_t pixel_value;
  };

  int num = 0;
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (pt->is_normal_initialized_)
    {
      Feature *ref_ftr;
      ref_ftr = pt->ref_patch;
      // Feature* ref_ftr;
      V2D pc(new_frame_->w2c(pt->pos_));
      V2D pc_prior(new_frame_->w2c_prior(pt->pos_));

      V3D norm_vec(ref_ftr->T_f_w_.rotationMatrix() * pt->normal_);
      V3D pf(ref_ftr->T_f_w_ * pt->pos_);

      if (pf.dot(norm_vec) < 0) norm_vec = -norm_vec;

      // norm_vec << norm_vec(1), norm_vec(0), norm_vec(2);
      cv::Mat img_cur = new_frame_->img_;
      cv::Mat img_ref = ref_ftr->img_;

      SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();
      Matrix2d A_cur_ref;
      getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref);

      // const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
      int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

      double D = A_cur_ref.determinant();
      if (D > 3) continue;

      num++;

      cv::Mat ref_cur_combine_temp;
      int radius = 20;
      cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
      cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

      getImagePatch(img_cur, pc, patch_buffer.data(), 0);

      float error_est = 0.0;
      float error_gt = 0.0;

      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error_est += (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]) *
                     (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]);
      }
      std::string ref_est = "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
      std::string cur_est = "cur_est " + std::to_string(1.0 / state->inv_expo_time);
      std::string cur_propa = "cur_gt " + std::to_string(error_gt);
      std::string cur_optimize = "cur_est " + std::to_string(error_est);

      cv::putText(ref_cur_combine_temp, ref_est, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40, ref_ftr->px_[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4,
                  cv::Scalar(0, 255, 0), 1, 8, 0);

      cv::putText(ref_cur_combine_temp, cur_est, cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_propa, cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, 8,
                  0);
      cv::putText(ref_cur_combine_temp, cur_optimize, cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8,
                  0);

      cv::rectangle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius, ref_ftr->px_[1] - radius),
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius, ref_ftr->px_[1] + radius), cv::Scalar(0, 0, 255), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc[0] - radius, pc[1] - radius), cv::Point2f(pc[0] + radius, pc[1] + radius),
                    cv::Scalar(0, 255, 0), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                    cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius), cv::Scalar(255, 255, 255), 1);
      cv::circle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]), 1, cv::Scalar(0, 0, 255), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1, cv::Scalar(0, 255, 0), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1, cv::Scalar(255, 255, 255), -1, 8);
      cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" + std::to_string(ref_ftr->id_) + "_" + std::to_string(num) + ".png",
                  ref_cur_combine_temp);

      std::vector<std::vector<pixel_member>> pixel_warp_matrix;

      for (int y = 0; y < patch_size; ++y)
      {
        vector<pixel_member> pixel_warp_vec;
        for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
        {
          Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
          px_patch *= (1 << search_level);
          const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
          uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

          const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
          if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
            continue;
          else
          {
            pixel_member pixel_warp;
            pixel_warp.pixel_pos << px[0], px[1];
            pixel_warp.pixel_value = pixel_value;
            pixel_warp_vec.push_back(pixel_warp);
          }
        }
        pixel_warp_matrix.push_back(pixel_warp_vec);
      }

      float x_min = 1000;
      float y_min = 1000;
      float x_max = 0;
      float y_max = 0;

      for (int i = 0; i < pixel_warp_matrix.size(); i++)
      {
        vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
        for (int j = 0; j < pixel_warp_row.size(); j++)
        {
          float x_temp = pixel_warp_row[j].pixel_pos[0];
          float y_temp = pixel_warp_row[j].pixel_pos[1];
          if (x_temp < x_min) x_min = x_temp;
          if (y_temp < y_min) y_min = y_temp;
          if (x_temp > x_max) x_max = x_temp;
          if (y_temp > y_max) y_max = y_temp;
        }
      }
      int x_min_i = floor(x_min);
      int y_min_i = floor(y_min);
      int x_max_i = ceil(x_max);
      int y_max_i = ceil(y_max);
      Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();
      for (int i = x_min_i; i < x_max_i; i++)
      {
        for (int j = y_min_i; j < y_max_i; j++)
        {
          Eigen::Vector2f pc_temp(i, j);
          Vector2f px_patch = A_cur_ref_Inv * (pc_temp - pc.cast<float>());
          if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) && px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
              px_patch[1] > (-patch_size / 2 * (1 << search_level)) && px_patch[1] < (patch_size / 2 * (1 << search_level)))
          {
            const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
            uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
            it_normal[width * j + i] = pixel_value;
          }
        }
      }
    }
  }
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;

    Feature *ref_ftr;
    V2D pc(new_frame_->w2c(pt->pos_));
    ref_ftr = pt->ref_patch;

    Matrix2d A_cur_ref;
    getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0,
                        patch_size_half, A_cur_ref);
    int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);
    double D = A_cur_ref.determinant();
    if (D > 3) continue;

    cv::Mat img_cur = new_frame_->img_;
    cv::Mat img_ref = ref_ftr->img_;
    for (int y = 0; y < patch_size; ++y)
    {
      for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
      {
        Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
        px_patch *= (1 << search_level);
        const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
        uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

        const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
        if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
          continue;
        else
        {
          int col = int(px[0]);
          int row = int(px[1]);
          it[width * row + col] = pixel_value;
        }
      }
    }
  }
  cv::Mat ref_cur_combine;
  cv::Mat ref_cur_combine_normal;
  cv::Mat ref_cur_combine_error;

  cv::hconcat(result, new_frame_->img_, ref_cur_combine);
  cv::hconcat(result_normal, new_frame_->img_, ref_cur_combine_normal);

  cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
  cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
  cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
  cv::hconcat(img_photometric_error, new_frame_->img_, ref_cur_combine_error);

  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png", ref_cur_combine);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                  "photometric"
                  ".png",
              ref_cur_combine_error);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png", ref_cur_combine_normal);
}

void VIOManager::precomputeReferencePatches(int level)
{
  double t1 = omp_get_wtime();
  if (total_points == 0) return;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;

  const int H_DIM = total_points * patch_size_total;

  H_sub_inv.resize(H_DIM, 6);
  H_sub_inv.setZero();
  M3D p_w_hat;

  for (int i = 0; i < total_points; i++)
  {
    const int scale = (1 << level);

    VisualPoint *pt = visual_submap->voxel_points[i];
    cv::Mat img = pt->ref_patch->img_;

    if (pt == nullptr) continue;

    double depth((pt->pos_ - pt->ref_patch->pos()).norm());
    V3D pf = pt->ref_patch->f_ * depth;
    V2D pc = pt->ref_patch->px_;
    M3D R_ref_w = pt->ref_patch->T_f_w_.rotationMatrix();

    computeProjectionJacobian(pf, Jdpi);
    p_w_hat << SKEW_SYM_MATRX(pt->pos_);

    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    for (int x = 0; x < patch_size; x++)
    {
      uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
      for (int y = 0; y < patch_size; ++y, img_ptr += scale)
      {
        float du =
            0.5f *
            ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
              w_ref_br * img_ptr[scale * width + scale * 2]) -
             (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
        float dv =
            0.5f *
            ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
              w_ref_br * img_ptr[width * scale * 2 + scale]) -
             (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

        Jimg << du, dv;
        Jimg = Jimg * (1.0 / scale);

        JdR = Jimg * Jdpi * R_ref_w * p_w_hat;
        Jdt = -Jimg * Jdpi * R_ref_w;

        H_sub_inv.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
      }
    }
  }
  has_ref_patch_cache = true;
}

void VIOManager::updateStateInverse(cv::Mat img, int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);
  V2D pc;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;
  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();
  compute_jacobian_time = update_ekf_time = 0.0;
  M3D P_wi_hat;
  bool z_init = true;
  const int H_DIM = total_points * patch_size_total;

  z.resize(H_DIM);
  z.setZero();

  H_sub.resize(H_DIM, 6);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();
    double count_outlier = 0;
    if (has_ref_patch_cache == false) precomputeReferencePatches(level);
    int n_meas = 0;
    float error = 0.0;
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;

    M3D p_hat;

    for (int i = 0; i < total_points; i++)
    {
      float patch_error = 0.0;

      const int scale = (1 << level);

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      pc = cam->world2cam(pf);

      const float u_ref = pc[0];
      const float v_ref = pc[1];
      const int u_ref_i = floorf(pc[0] / scale) * scale;
      const int v_ref_i = floorf(pc[1] / scale) * scale;
      const float subpix_u_ref = (u_ref - u_ref_i) / scale;
      const float subpix_v_ref = (v_ref - v_ref_i) / scale;
      const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      const float w_ref_br = subpix_u_ref * subpix_v_ref;

      const vector<float> &P = visual_submap->warp_patch[i];
      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                       w_ref_br * img_ptr[scale * width + scale] - P[patch_size_total * level + x * patch_size + y];
          z(i * patch_size_total + x * patch_size + y) = res;
          patch_error += res * res;
          MD(1, 3) J_dR = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 0);
          MD(1, 3) J_dt = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 3);
          JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
          Jdt = J_dt * Rwi;
          H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
          n_meas++;
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
      Eigen::Matrix<double, 6, 6> pose_info = H_T_H.block<6, 6>(0, 0);
      Eigen::Matrix<double, 6, 1> pose_rhs = H_sub_T * z;
      if (saif_gate_en)
      {
        const SaifGateStats saif_stats = applySaifGate(pose_info, pose_rhs, saif_min_sqrt_info, saif_min_weight);
        H_T_H.block<6, 6>(0, 0) = pose_info;
        saif_gated_dirs += saif_stats.gated_dirs;
        saif_frame_min_weight = std::min(saif_frame_min_weight, saif_stats.min_weight);
        saif_frame_weight_sum += saif_stats.weight_sum / 6.0;
        saif_frame_samples++;
      }
      auto vec = (*state_propagat) - (*state);
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      auto solution = -K_1.block<DIM_STATE, 6>(0, 0) * pose_rhs + vec - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f)) { EKF_end = true; }
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break; 
  }
}

void VIOManager::updateState(cv::Mat img, int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);

  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  const int H_DIM = total_points * patch_size_total;
  z.resize(H_DIM);
  z.setZero();
  H_sub.resize(H_DIM, 7);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();

    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    Jdp_dt = Rci * Rwi.transpose();
    
    float error = 0.0;
    int n_meas = 0;
    // int max_threads = omp_get_max_threads();
    // int desired_threads = std::min(max_threads, total_points);
    // omp_set_num_threads(desired_threads);
  
    #ifdef MP_EN
      omp_set_num_threads(MP_PROC_NUM);
      #pragma omp parallel for reduction(+:error, n_meas)
    #endif
    for (int i = 0; i < total_points; i++)
    {
      // printf("thread is %d, i=%d, i address is %p\n", omp_get_thread_num(), i, &i);
      MD(1, 2) Jimg;
      MD(2, 3) Jdpi;
      MD(1, 3) Jdphi, Jdp, JdR, Jdt;

      float patch_error = 0.0;
      int search_level = visual_submap->search_levels[i];
      int pyramid_level = level + search_level;
      int scale = (1 << pyramid_level);
      float inv_scale = 1.0f / scale;

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      V2D pc = cam->world2cam(pf);

      computeProjectionJacobian(pf, Jdpi);
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);

      float u_ref = pc[0];
      float v_ref = pc[1];
      int u_ref_i = floorf(pc[0] / scale) * scale;
      int v_ref_i = floorf(pc[1] / scale) * scale;
      float subpix_u_ref = (u_ref - u_ref_i) / scale;
      float subpix_v_ref = (v_ref - v_ref_i) / scale;
      float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      float w_ref_br = subpix_u_ref * subpix_v_ref;

      const vector<float> &P = visual_submap->warp_patch[i];
      double inv_ref_expo = visual_submap->inv_expo_list[i];
      // ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n", inv_ref_expo, state->inv_expo_time);

      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          float du =
              0.5f *
              ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
                w_ref_br * img_ptr[scale * width + scale * 2]) -
               (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
          float dv =
              0.5f *
              ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
                w_ref_br * img_ptr[width * scale * 2 + scale]) -
               (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

          Jimg << du, dv;
          Jimg = Jimg * state->inv_expo_time;
          Jimg = Jimg * inv_scale;
          Jdphi = Jimg * Jdpi * p_hat;
          Jdp = -Jimg * Jdpi;
          JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          Jdt = Jdp * Jdp_dt;

          double cur_value =
              w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
          double res = state->inv_expo_time * cur_value - inv_ref_expo * P[patch_size_total * level + x * patch_size + y];

          z(i * patch_size_total + x * patch_size + y) = res;

          patch_error += res * res;
          n_meas += 1;
          
          if (exposure_estimate_en) { H_sub.block<1, 7>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt, cur_value; }
          else { H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt; }
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;
    
    compute_jacobian_time += omp_get_wtime() - t1;

    // printf("\nPYRAMID LEVEL %i\n---------------\n", level);
    // std::cout << "It. " << iteration
    //           << "\t last_error = " << last_error
    //           << "\t new_error = " << error
    //           << std::endl;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov; auto
      // vec = (*state_propagat) - (*state); G = K*H;
      // (*state) += (-K*z + vec - G*vec);

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 7>(0, 0) = K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
      MD(DIM_STATE, 1)
      solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec - G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);

      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      auto &&expo_add = solution.block<1, 1>(6, 0);
      // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))  EKF_end = true;
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
  // if (state->inv_expo_time < 0.0)  {ROS_ERROR("reset expo time!!!!!!!!!!\n"); state->inv_expo_time = 0.0;}
}

void VIOManager::updateFrameState(StatesGroup state)
{
  M3D Rwi(state.rot_end);
  V3D Pwi(state.pos_end);
  Rcw = NormalizeRotation(Rci * Rwi.transpose());
  Pcw = -Rcw * Pwi + Pci;
  new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}

void VIOManager::plotTrackedPoints()
{
  int total_points = visual_submap->voxel_points.size();
  if (total_points == 0) return;
  // int inlier_count = 0;
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Poaint2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    V2D pc(new_frame_->w2c(pt->pos_));

    if (visual_submap->errors[i] <= visual_submap->propa_errors[i])
    {
      // inlier_count++;
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
    }
    else
    {
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }
  }
  // std::string text = std::to_string(inlier_count) + " " + std::to_string(total_points);
  // cv::Point2f origin;
  // origin.x = img_cp.cols - 110;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, 8, 0);
}

V3F VIOManager::getInterpolatedPixel(cv::Mat img, V2D pc)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int u_ref_i = floorf(pc[0]);
  const int v_ref_i = floorf(pc[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
  V3F pixel(B, G, R);
  return pixel;
}

void VIOManager::dumpDataForColmap()
{
  static int cnt = 1;
  std::ostringstream ss;
  ss << std::setw(5) << std::setfill('0') << cnt;
  std::string cnt_str = ss.str();
  std::string image_path = std::string(ROOT_DIR) + "Log/Colmap/images/" + cnt_str + ".png";
  
  cv::Mat img_rgb_undistort;
  pinhole_cam->undistortImage(img_rgb, img_rgb_undistort);
  cv::imwrite(image_path, img_rgb_undistort);
  
  Eigen::Quaterniond q(new_frame_->T_f_w_.rotationMatrix());
  Eigen::Vector3d t = new_frame_->T_f_w_.translation();
  fout_colmap << cnt << " "
            << std::fixed << std::setprecision(6)  // 保证浮点数精度为6位
            << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
            << t.x() << " " << t.y() << " " << t.z() << " "
            << 1 << " "  // CAMERA_ID (假设相机ID为1)
            << cnt_str << ".png" << std::endl;
  fout_colmap << "0.0 0.0 -1" << std::endl;
  cnt++;
}

void VIOManager::buildFeatureVioLandmarks(const cv::Mat &img, const vector<pointWithVar> &pg, vector<FeatureVioLandmark> &landmarks)
{
  const double detect_t0 = omp_get_wtime();
  landmarks.clear();
  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(img, corners, feature_vio_max_features, feature_vio_quality_level, feature_vio_min_distance,
                          cv::Mat(), 3, false, 0.04);
  feature_vio_detect_time = omp_get_wtime() - detect_t0;
  feature_vio_new_features = corners.size();
  if (corners.empty() || pg.empty()) return;

  const double project_t0 = omp_get_wtime();
  cv::Mat index_img;
  if (feature_vio_depth_image_reuse_en)
  {
    feature_vio_index_img_.create(height, width, CV_32SC1);
    feature_vio_index_img_.setTo(-1);
    index_img = feature_vio_index_img_;
  }
  else
  {
    index_img = cv::Mat(height, width, CV_32SC1, cv::Scalar(-1));
  }
  std::vector<V3D> projected_points;
  std::vector<float> projected_depths;
  projected_points.reserve(pg.size());
  projected_depths.reserve(pg.size());

  for (const pointWithVar &pv : pg)
  {
    const V3D pf = new_frame_->w2f(pv.point_w);
    if (pf[2] < feature_vio_min_depth || pf[2] > feature_vio_max_depth) continue;
    const V2D px = new_frame_->cam_->world2cam(pf);
    if (!new_frame_->cam_->isInFrame(px.cast<int>(), border)) continue;

    const int u = static_cast<int>(std::round(px[0]));
    const int v = static_cast<int>(std::round(px[1]));
    if (u < 0 || u >= width || v < 0 || v >= height) continue;

    int &point_index = index_img.at<int>(v, u);
    if (point_index < 0 || pf[2] < projected_depths[point_index])
    {
      point_index = static_cast<int>(projected_points.size());
      projected_points.push_back(pv.point_w);
      projected_depths.push_back(static_cast<float>(pf[2]));
    }
  }
  feature_vio_depth_project_time = omp_get_wtime() - project_t0;

  const double match_t0 = omp_get_wtime();
  const int radius = std::max(0, feature_vio_depth_search_radius);
  landmarks.reserve(corners.size());
  for (const cv::Point2f &corner : corners)
  {
    const int center_u = static_cast<int>(std::round(corner.x));
    const int center_v = static_cast<int>(std::round(corner.y));
    int best_idx = -1;
    int best_dist_sq = std::numeric_limits<int>::max();
    float best_depth = std::numeric_limits<float>::max();

    for (int dv = -radius; dv <= radius; ++dv)
    {
      const int v = center_v + dv;
      if (v < 0 || v >= height) continue;
      for (int du = -radius; du <= radius; ++du)
      {
        const int u = center_u + du;
        if (u < 0 || u >= width) continue;
        const int idx = index_img.at<int>(v, u);
        if (idx < 0) continue;
        const int dist_sq = du * du + dv * dv;
        const float depth = projected_depths[idx];
        if (dist_sq < best_dist_sq || (dist_sq == best_dist_sq && depth < best_depth))
        {
          best_dist_sq = dist_sq;
          best_depth = depth;
          best_idx = idx;
        }
      }
    }

    if (best_idx >= 0)
    {
      FeatureVioLandmark landmark;
      landmark.px = corner;
      landmark.point_w = projected_points[best_idx];
      landmark.depth = best_depth;
      landmark.depth_px_dist = std::sqrt(static_cast<double>(best_dist_sq));
      landmarks.push_back(landmark);
    }
  }
  feature_vio_depth_landmarks = landmarks.size();
  feature_vio_depth_match_time = omp_get_wtime() - match_t0;
}

void VIOManager::runFeatureVioDiagnostic(const cv::Mat &img, const vector<pointWithVar> &pg)
{
  const double t0 = omp_get_wtime();
  feature_vio_prev_landmarks = feature_vio_prev_landmarks_.size();
  feature_vio_tracked = 0;
  feature_vio_valid_reproj = 0;
  feature_vio_inliers = 0;
  feature_vio_new_features = 0;
  feature_vio_depth_landmarks = 0;
  feature_vio_fb_inliers = 0;
  feature_vio_fb_rejects = 0;
  feature_vio_fb_check_run = false;
  feature_vio_track_time = 0.0;
  feature_vio_solve_time = 0.0;
  feature_vio_landmark_time = 0.0;
  feature_vio_detect_time = 0.0;
  feature_vio_depth_project_time = 0.0;
  feature_vio_depth_match_time = 0.0;
  feature_vio_ransac_inliers = 0;
  feature_vio_robust_inliers = 0;
  feature_vio_reproj_rmse = 0.0;
  feature_vio_inlier_ratio = 0.0;
  feature_vio_robust_reproj_rmse = 0.0;
  feature_vio_robust_inlier_ratio = 0.0;
  feature_vio_robust_depth_mean = 0.0;
  feature_vio_robust_depth_min = 0.0;
  feature_vio_robust_depth_max = 0.0;
  feature_vio_robust_ref_depth_mean = 0.0;
  feature_vio_robust_depth_px_dist_mean = 0.0;
  feature_vio_robust_reproj_bias_u = 0.0;
  feature_vio_robust_reproj_bias_v = 0.0;
  feature_vio_robust_abs_reproj_mean = 0.0;
  feature_vio_mean_parallax = 0.0;
  feature_vio_solve_points = 0;
  feature_vio_dry_run_update_norm = 0.0;
  feature_vio_dry_run_rot_deg = 0.0;
  feature_vio_dry_run_pos_norm = 0.0;
  feature_vio_dry_run_condition = 0.0;
  feature_vio_dry_run_eig_min = 0.0;
  feature_vio_dry_run_eig_max = 0.0;
  feature_vio_weight_mean = 1.0;
  feature_vio_weight_min = 1.0;
  feature_vio_effective_points = 0.0;
  feature_vio_adaptive_depth_scene_mean = 0.0;
  feature_vio_adaptive_inlier_scene_mean = 0.0;
  feature_vio_commit_update_norm = 0.0;
  feature_vio_commit_rot_deg = 0.0;
  feature_vio_commit_pos_norm = 0.0;
  feature_vio_commit_rot_x = 0.0;
  feature_vio_commit_rot_y = 0.0;
  feature_vio_commit_rot_z = 0.0;
  feature_vio_commit_pos_x = 0.0;
  feature_vio_commit_pos_y = 0.0;
  feature_vio_commit_pos_z = 0.0;
  feature_vio_lio_weak_dir = lio_info_valid ? lio_info_weak_dir : -1;
  feature_vio_lio_weak_abs_projection = 0.0;
  feature_vio_lio_weak_abs_cosine = 0.0;
  feature_vio_bias_watchdog_samples = feature_vio_bias_watchdog_window_.size();
  feature_vio_bias_watchdog_rot_mean_norm = 0.0;
  feature_vio_bias_watchdog_pos_mean_norm = 0.0;
  feature_vio_bias_watchdog_reject = false;
  feature_vio_saif_gated_dirs = 0;
  feature_vio_saif_min_weight = 1.0;
  feature_vio_saif_weight_avg = 1.0;
  feature_vio_gate_pass = false;
  feature_vio_commit_pass = false;
  feature_vio_dry_run_status = feature_vio_dry_run_en ? 2 : 0;
  feature_vio_commit_status = feature_vio_update_en ? 3 : 2;

  const int lk_window = std::max(3, feature_vio_lk_window_size | 1);
  const int lk_max_level = std::max(0, feature_vio_lk_max_level);
  const cv::Size lk_window_size(lk_window, lk_window);
  std::vector<cv::Mat> cur_pyramid;
  bool temporal_pyramid_updated = false;

  if (!feature_vio_prev_img_.empty() && !feature_vio_prev_landmarks_.empty())
  {
    std::vector<cv::Point2f> prev_px;
    prev_px.reserve(feature_vio_prev_landmarks_.size());
    for (const FeatureVioLandmark &landmark : feature_vio_prev_landmarks_)
    {
      prev_px.push_back(landmark.px);
    }

    std::vector<cv::Point2f> cur_px;
    std::vector<uchar> status;
    std::vector<float> tracking_error;
    std::vector<cv::Mat> prev_pyramid;
    if (feature_vio_lk_pyramid_cache_en)
    {
      const bool temporal_cache_valid = feature_vio_lk_temporal_cache_en && !feature_vio_prev_lk_pyramid_.empty() &&
                                        feature_vio_prev_lk_window_ == lk_window &&
                                        feature_vio_prev_lk_max_level_ == lk_max_level;
      if (feature_vio_lk_temporal_cache_en)
      {
        cv::buildOpticalFlowPyramid(img, cur_pyramid, lk_window_size, lk_max_level, true, cv::BORDER_REFLECT_101,
                                    cv::BORDER_CONSTANT, false);
      }
      else
      {
        cv::buildOpticalFlowPyramid(img, cur_pyramid, lk_window_size, lk_max_level, true);
      }
      const std::vector<cv::Mat> *tracking_prev_pyramid = &feature_vio_prev_lk_pyramid_;
      if (!temporal_cache_valid)
      {
        if (feature_vio_lk_temporal_cache_en)
        {
          cv::buildOpticalFlowPyramid(feature_vio_prev_img_, prev_pyramid, lk_window_size, lk_max_level, false,
                                      cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
        }
        else
        {
          cv::buildOpticalFlowPyramid(feature_vio_prev_img_, prev_pyramid, lk_window_size, lk_max_level, true);
        }
        tracking_prev_pyramid = &prev_pyramid;
      }
      cv::calcOpticalFlowPyrLK(*tracking_prev_pyramid, cur_pyramid, prev_px, cur_px, status, tracking_error,
                               lk_window_size, lk_max_level);
    }
    else
    {
      cv::calcOpticalFlowPyrLK(feature_vio_prev_img_, img, prev_px, cur_px, status, tracking_error, lk_window_size, lk_max_level);
    }

    feature_vio_fb_check_run = feature_vio_fb_check_en;

    std::vector<cv::Point2f> back_px;
    std::vector<uchar> back_status;
    std::vector<float> back_tracking_error;
    if (feature_vio_fb_check_run && !cur_px.empty())
    {
      if (feature_vio_lk_pyramid_cache_en)
      {
        const bool temporal_cache_valid = feature_vio_lk_temporal_cache_en && !feature_vio_prev_lk_pyramid_.empty() &&
                                          feature_vio_prev_lk_window_ == lk_window &&
                                          feature_vio_prev_lk_max_level_ == lk_max_level;
        const std::vector<cv::Mat> &tracking_prev_pyramid = temporal_cache_valid ? feature_vio_prev_lk_pyramid_ : prev_pyramid;
        cv::calcOpticalFlowPyrLK(cur_pyramid, tracking_prev_pyramid, cur_px, back_px, back_status, back_tracking_error,
                                 lk_window_size, lk_max_level);
      }
      else
      {
        cv::calcOpticalFlowPyrLK(img, feature_vio_prev_img_, cur_px, back_px, back_status, back_tracking_error, lk_window_size, lk_max_level);
      }
    }
    if (feature_vio_lk_pyramid_cache_en && feature_vio_lk_temporal_cache_en)
    {
      feature_vio_prev_lk_pyramid_.clear();
      feature_vio_prev_lk_pyramid_.reserve((cur_pyramid.size() + 1) / 2);
      for (size_t level_idx = 0; level_idx < cur_pyramid.size(); level_idx += 2)
      {
        feature_vio_prev_lk_pyramid_.push_back(cur_pyramid[level_idx]);
      }
      cur_pyramid.clear();
      feature_vio_prev_lk_window_ = lk_window;
      feature_vio_prev_lk_max_level_ = lk_max_level;
      temporal_pyramid_updated = true;
    }
    feature_vio_track_time = omp_get_wtime() - t0;

    struct FeatureVioCandidate
    {
      cv::Point2f prev_px;
      cv::Point2f cur_px;
      V3D point_w = V3D::Zero();
      double reproj_error_sq = 0.0;
      double reproj_error_u = 0.0;
      double reproj_error_v = 0.0;
      double ref_depth = 0.0;
      double cur_depth = 0.0;
      double depth_px_dist = 0.0;
    };
    std::vector<FeatureVioCandidate> candidates;
    std::vector<cv::Point2f> ransac_prev_px;
    std::vector<cv::Point2f> ransac_cur_px;
    candidates.reserve(cur_px.size());
    ransac_prev_px.reserve(cur_px.size());
    ransac_cur_px.reserve(cur_px.size());

    double reproj_error_sq_sum = 0.0;
    double parallax_sum = 0.0;
    for (size_t i = 0; i < cur_px.size(); ++i)
    {
      if (!status[i]) continue;
      const cv::Point2f &tracked_px = cur_px[i];
      if (tracked_px.x < border || tracked_px.x >= width - border || tracked_px.y < border || tracked_px.y >= height - border) continue;
      feature_vio_tracked++;
      const cv::Point2f delta = tracked_px - prev_px[i];
      parallax_sum += std::sqrt(delta.x * delta.x + delta.y * delta.y);

      const V3D pf = new_frame_->w2f(feature_vio_prev_landmarks_[i].point_w);
      if (pf[2] <= feature_vio_min_depth || pf[2] > feature_vio_max_depth) continue;
      const V2D reproj = new_frame_->w2c(feature_vio_prev_landmarks_[i].point_w);
      if (!new_frame_->cam_->isInFrame(reproj.cast<int>(), border)) continue;

      const double du = tracked_px.x - reproj[0];
      const double dv = tracked_px.y - reproj[1];
      const double err_sq = du * du + dv * dv;
      reproj_error_sq_sum += err_sq;
      feature_vio_valid_reproj++;
      if (std::sqrt(err_sq) <= feature_vio_inlier_thresh_px) feature_vio_inliers++;

      bool fb_ok = true;
      if (feature_vio_fb_check_run)
      {
        fb_ok = false;
        if (i < back_status.size() && back_status[i])
        {
          const cv::Point2f fb_delta = back_px[i] - prev_px[i];
          const double fb_err = std::sqrt(fb_delta.x * fb_delta.x + fb_delta.y * fb_delta.y);
          fb_ok = fb_err <= feature_vio_fb_thresh_px;
        }
      }

      if (fb_ok)
      {
        feature_vio_fb_inliers++;
        FeatureVioCandidate candidate;
        candidate.prev_px = prev_px[i];
        candidate.cur_px = tracked_px;
        candidate.point_w = feature_vio_prev_landmarks_[i].point_w;
        candidate.reproj_error_sq = err_sq;
        candidate.reproj_error_u = du;
        candidate.reproj_error_v = dv;
        candidate.ref_depth = feature_vio_prev_landmarks_[i].depth;
        candidate.cur_depth = pf[2];
        candidate.depth_px_dist = feature_vio_prev_landmarks_[i].depth_px_dist;
        candidates.push_back(candidate);
        ransac_prev_px.push_back(prev_px[i]);
        ransac_cur_px.push_back(tracked_px);
      }
      else
      {
        feature_vio_fb_rejects++;
      }
    }

    if (feature_vio_valid_reproj > 0)
    {
      feature_vio_reproj_rmse = std::sqrt(reproj_error_sq_sum / static_cast<double>(feature_vio_valid_reproj));
      feature_vio_inlier_ratio = static_cast<double>(feature_vio_inliers) / static_cast<double>(feature_vio_valid_reproj);
    }
    if (feature_vio_tracked > 0)
    {
      feature_vio_mean_parallax = parallax_sum / static_cast<double>(feature_vio_tracked);
    }

    std::vector<uchar> ransac_mask(candidates.size(), 1);
    if (feature_vio_ransac_en)
    {
      ransac_mask.assign(candidates.size(), 0);
      if (candidates.size() >= 8)
      {
        cv::Mat ransac_cv_mask;
        const cv::Mat fundamental = cv::findFundamentalMat(ransac_prev_px, ransac_cur_px, cv::FM_RANSAC, feature_vio_ransac_thresh_px, 0.99, ransac_cv_mask);
        if (!fundamental.empty() && static_cast<size_t>(ransac_cv_mask.rows) == candidates.size())
        {
          for (size_t i = 0; i < candidates.size(); ++i)
          {
            ransac_mask[i] = ransac_cv_mask.at<uchar>(static_cast<int>(i), 0) != 0;
          }
        }
      }
    }

    double robust_reproj_error_sq_sum = 0.0;
    double robust_reproj_error_sum_u = 0.0;
    double robust_reproj_error_sum_v = 0.0;
    double robust_abs_reproj_sum = 0.0;
    double robust_depth_sum = 0.0;
    double robust_ref_depth_sum = 0.0;
    double robust_depth_px_dist_sum = 0.0;
    double robust_depth_min = std::numeric_limits<double>::max();
    double robust_depth_max = 0.0;
    std::vector<FeatureVioCandidate> robust_candidates;
    robust_candidates.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i)
    {
      if (!ransac_mask[i]) continue;
      feature_vio_ransac_inliers++;
      if (std::sqrt(candidates[i].reproj_error_sq) > feature_vio_inlier_thresh_px) continue;
      robust_reproj_error_sq_sum += candidates[i].reproj_error_sq;
      robust_reproj_error_sum_u += candidates[i].reproj_error_u;
      robust_reproj_error_sum_v += candidates[i].reproj_error_v;
      robust_abs_reproj_sum += std::sqrt(candidates[i].reproj_error_sq);
      robust_depth_sum += candidates[i].cur_depth;
      robust_ref_depth_sum += candidates[i].ref_depth;
      robust_depth_px_dist_sum += candidates[i].depth_px_dist;
      robust_depth_min = std::min(robust_depth_min, candidates[i].cur_depth);
      robust_depth_max = std::max(robust_depth_max, candidates[i].cur_depth);
      feature_vio_robust_inliers++;
      robust_candidates.push_back(candidates[i]);
    }
    if (feature_vio_valid_reproj > 0)
    {
      feature_vio_robust_inlier_ratio = static_cast<double>(feature_vio_robust_inliers) / static_cast<double>(feature_vio_valid_reproj);
    }
    if (feature_vio_robust_inliers > 0)
    {
      const double inv_robust_count = 1.0 / static_cast<double>(feature_vio_robust_inliers);
      feature_vio_robust_reproj_rmse = std::sqrt(robust_reproj_error_sq_sum * inv_robust_count);
      feature_vio_robust_depth_mean = robust_depth_sum * inv_robust_count;
      feature_vio_robust_depth_min = robust_depth_min;
      feature_vio_robust_depth_max = robust_depth_max;
      feature_vio_robust_ref_depth_mean = robust_ref_depth_sum * inv_robust_count;
      feature_vio_robust_depth_px_dist_mean = robust_depth_px_dist_sum * inv_robust_count;
      feature_vio_robust_reproj_bias_u = robust_reproj_error_sum_u * inv_robust_count;
      feature_vio_robust_reproj_bias_v = robust_reproj_error_sum_v * inv_robust_count;
      feature_vio_robust_abs_reproj_mean = robust_abs_reproj_sum * inv_robust_count;
    }
    feature_vio_gate_pass = feature_vio_robust_inliers >= static_cast<size_t>(std::max(0, feature_vio_gate_min_inliers)) &&
                            feature_vio_robust_inlier_ratio >= feature_vio_gate_min_inlier_ratio;
    if (feature_vio_gate_pass && feature_vio_robust_inliers > 0)
    {
      feature_vio_adaptive_depth_scene_sum_ += feature_vio_robust_depth_px_dist_mean;
      feature_vio_adaptive_inlier_scene_sum_ += feature_vio_robust_inlier_ratio;
      feature_vio_adaptive_depth_scene_samples_++;
    }
    if (feature_vio_adaptive_depth_scene_samples_ > 0)
    {
      const double scene_samples = static_cast<double>(feature_vio_adaptive_depth_scene_samples_);
      feature_vio_adaptive_depth_scene_mean = feature_vio_adaptive_depth_scene_sum_ / scene_samples;
      feature_vio_adaptive_inlier_scene_mean = feature_vio_adaptive_inlier_scene_sum_ / scene_samples;
    }

    if (feature_vio_dry_run_en)
    {
      if (!feature_vio_gate_pass)
      {
        feature_vio_dry_run_status = 2;
      }
      else if (robust_candidates.size() < 6)
      {
        feature_vio_dry_run_status = 3;
      }
      else
      {
        Eigen::Matrix<double, 6, 6> HtH = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> Htr = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 1> solution = Eigen::Matrix<double, 6, 1>::Zero();
        M3D Rwi(state->rot_end);
        V3D Pwi(state->pos_end);
        Rcw = Rci * Rwi.transpose();
        Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
        Jdp_dt = Rci * Rwi.transpose();
        double weight_sum = 0.0;
        double min_weight = 1.0;

        for (const FeatureVioCandidate &candidate : robust_candidates)
        {
          const V3D pf = Rcw * candidate.point_w + Pcw;
          if (pf[2] <= feature_vio_min_depth || pf[2] > feature_vio_max_depth) continue;
          feature_vio_solve_points++;
          const V2D projected_px = cam->world2cam(pf);
          MD(2, 3) Jdpi;
          computeProjectionJacobian(pf, Jdpi);
          M3D p_hat;
          p_hat << SKEW_SYM_MATRX(pf);
          const MD(2, 3) Jdphi = Jdpi * p_hat;
          const MD(2, 3) Jdp = -Jdpi;
          const MD(2, 3) JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          const MD(2, 3) Jdt = Jdp * Jdp_dt;
          Eigen::Matrix<double, 2, 6> H_i;
          H_i << JdR, Jdt;
          Eigen::Matrix<double, 2, 1> residual;
          residual << projected_px[0] - candidate.cur_px.x, projected_px[1] - candidate.cur_px.y;
          double weight = 1.0;
          const bool adaptive_scene_gate_pass = feature_vio_adaptive_depth_scene_gate_px > 0.0 &&
                                                feature_vio_adaptive_depth_scene_mean >= feature_vio_adaptive_depth_scene_gate_px;
          const bool adaptive_frame_gate_pass = feature_vio_adaptive_depth_gate_px <= 0.0 ||
                                                feature_vio_robust_depth_px_dist_mean >= feature_vio_adaptive_depth_gate_px;
          const bool adaptive_weight_run = feature_vio_adaptive_weight_en &&
                                           (adaptive_scene_gate_pass || adaptive_frame_gate_pass);
          if (adaptive_weight_run)
          {
            const double residual_norm = residual.norm();
            const double huber_px = std::max(feature_vio_adaptive_huber_px, std::numeric_limits<double>::epsilon());
            const double huber_weight = residual_norm <= huber_px ? 1.0 : huber_px / residual_norm;
            const double depth_sigma_px =
                std::max(feature_vio_adaptive_depth_sigma_px, std::numeric_limits<double>::epsilon());
            const double normalized_depth_distance = candidate.depth_px_dist / depth_sigma_px;
            const double depth_weight = std::exp(-0.5 * normalized_depth_distance * normalized_depth_distance);
            const double min_adaptive_weight = std::min(1.0, std::max(0.0, feature_vio_adaptive_min_weight));
            weight = std::max(min_adaptive_weight, huber_weight * depth_weight);
          }
          HtH.noalias() += weight * H_i.transpose() * H_i;
          Htr.noalias() += weight * H_i.transpose() * residual;
          weight_sum += weight;
          min_weight = std::min(min_weight, weight);
        }

        if (feature_vio_solve_points > 0)
        {
          feature_vio_weight_mean = weight_sum / static_cast<double>(feature_vio_solve_points);
          feature_vio_weight_min = min_weight;
          feature_vio_effective_points = weight_sum;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(HtH);
        if (eigensolver.info() != Eigen::Success)
        {
          feature_vio_dry_run_status = 4;
        }
        else
        {
          feature_vio_dry_run_eig_min = std::max(0.0, eigensolver.eigenvalues()[0]);
          feature_vio_dry_run_eig_max = std::max(0.0, eigensolver.eigenvalues()[5]);
          const double denom = std::max(feature_vio_dry_run_eig_min, feature_vio_dry_run_damping);
          feature_vio_dry_run_condition = feature_vio_dry_run_eig_max / denom;
          if (feature_vio_dry_run_condition > feature_vio_dry_run_max_condition || feature_vio_dry_run_eig_max <= 0.0)
          {
            feature_vio_dry_run_status = 4;
          }
          else
          {
            HtH.diagonal().array() += feature_vio_dry_run_damping;
            solution = -HtH.ldlt().solve(Htr);
            feature_vio_dry_run_update_norm = solution.norm();
            feature_vio_dry_run_rot_deg = solution.block<3, 1>(0, 0).norm() * 57.29577951308232;
            feature_vio_dry_run_pos_norm = solution.block<3, 1>(3, 0).norm();
            feature_vio_dry_run_status = 1;

            if (feature_vio_update_en)
            {
              Eigen::Matrix<double, 6, 1> commit_solution = solution;
              Eigen::Matrix<double, DIM_STATE, DIM_STATE> feature_gain = Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero();
              bool update_covariance = false;
              if (feature_vio_inekf_update_en)
              {
                Eigen::Matrix<double, 6, 6> pose_info = HtH;
                Eigen::Matrix<double, 6, 1> pose_rhs = Htr;
                if (saif_gate_en)
                {
                  const SaifGateStats saif_stats = applySaifGate(pose_info, pose_rhs, saif_min_sqrt_info, saif_min_weight);
                  feature_vio_saif_gated_dirs = saif_stats.gated_dirs;
                  feature_vio_saif_min_weight = saif_stats.min_weight;
                  feature_vio_saif_weight_avg = saif_stats.weight_sum / 6.0;
                }

                Eigen::Matrix<double, DIM_STATE, DIM_STATE> feature_info = Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero();
                feature_info.block<6, 6>(0, 0) = pose_info;
                const double pixel_cov = std::max(feature_vio_img_point_cov, std::numeric_limits<double>::epsilon());
                const Eigen::Matrix<double, DIM_STATE, DIM_STATE> prior_info = (state->cov / pixel_cov).inverse();
                const Eigen::Matrix<double, DIM_STATE, DIM_STATE> K_1 = (feature_info + prior_info).inverse();
                const Eigen::Matrix<double, DIM_STATE, 6> K_pose = K_1.block<DIM_STATE, 6>(0, 0);
                const Eigen::Matrix<double, DIM_STATE, 1> ekf_solution = -K_pose * pose_rhs;
                commit_solution = ekf_solution.block<6, 1>(0, 0);
                feature_gain.block<DIM_STATE, 6>(0, 0) = K_pose * pose_info;
                update_covariance = feature_vio_update_scale >= 0.0 && feature_vio_update_scale <= 1.0;
              }

              const Eigen::Matrix<double, 6, 1> scaled_solution = feature_vio_update_scale * commit_solution;
              feature_vio_commit_update_norm = scaled_solution.norm();
              feature_vio_commit_rot_deg = scaled_solution.block<3, 1>(0, 0).norm() * 57.29577951308232;
              feature_vio_commit_pos_norm = scaled_solution.block<3, 1>(3, 0).norm();
              feature_vio_commit_rot_x = scaled_solution[0];
              feature_vio_commit_rot_y = scaled_solution[1];
              feature_vio_commit_rot_z = scaled_solution[2];
              feature_vio_commit_pos_x = scaled_solution[3];
              feature_vio_commit_pos_y = scaled_solution[4];
              feature_vio_commit_pos_z = scaled_solution[5];
              if (lio_info_valid)
              {
                const double update_norm = std::max(feature_vio_commit_update_norm, std::numeric_limits<double>::epsilon());
                const double weak_norm = std::max(lio_info_weak_vec.norm(), std::numeric_limits<double>::epsilon());
                feature_vio_lio_weak_dir = lio_info_weak_dir;
                feature_vio_lio_weak_abs_projection = std::abs(scaled_solution.dot(lio_info_weak_vec) / weak_norm);
                feature_vio_lio_weak_abs_cosine = feature_vio_lio_weak_abs_projection / update_norm;
              }
              if (feature_vio_commit_update_norm > feature_vio_max_update_norm)
              {
                feature_vio_commit_status = 4;
              }
              else if (feature_vio_commit_rot_deg > feature_vio_max_rot_deg)
              {
                feature_vio_commit_status = 5;
              }
              else if (feature_vio_commit_pos_norm > feature_vio_max_pos_norm)
              {
                feature_vio_commit_status = 6;
              }
              else if (feature_vio_bias_watchdog_en)
              {
                const int window_size = std::max(1, feature_vio_bias_watchdog_window);
                feature_vio_bias_watchdog_window_.push_back(scaled_solution);
                feature_vio_bias_watchdog_sum_ += scaled_solution;
                while (static_cast<int>(feature_vio_bias_watchdog_window_.size()) > window_size)
                {
                  feature_vio_bias_watchdog_sum_ -= feature_vio_bias_watchdog_window_.front();
                  feature_vio_bias_watchdog_window_.pop_front();
                }

                feature_vio_bias_watchdog_samples = feature_vio_bias_watchdog_window_.size();
                const double inv_samples = 1.0 / static_cast<double>(feature_vio_bias_watchdog_samples);
                const Eigen::Matrix<double, 6, 1> mean_update = feature_vio_bias_watchdog_sum_ * inv_samples;
                feature_vio_bias_watchdog_rot_mean_norm = mean_update.block<3, 1>(0, 0).norm();
                feature_vio_bias_watchdog_pos_mean_norm = mean_update.block<3, 1>(3, 0).norm();
                const bool has_enough_samples =
                    feature_vio_bias_watchdog_samples >= static_cast<size_t>(std::max(1, feature_vio_bias_watchdog_min_samples));
                feature_vio_bias_watchdog_reject =
                    has_enough_samples &&
                    (feature_vio_bias_watchdog_rot_mean_norm > feature_vio_bias_watchdog_max_rot_mean ||
                     feature_vio_bias_watchdog_pos_mean_norm > feature_vio_bias_watchdog_max_pos_mean);
                if (feature_vio_bias_watchdog_reject)
                {
                  feature_vio_commit_status = 7;
                }
                else
                {
                  Eigen::Matrix<double, DIM_STATE, 1> state_update = Eigen::Matrix<double, DIM_STATE, 1>::Zero();
                  state_update.block<6, 1>(0, 0) = scaled_solution;
                  (*state) += state_update;
                  if (feature_vio_inekf_update_en && update_covariance)
                  {
                    const Eigen::Matrix<double, DIM_STATE, DIM_STATE> I_STATE =
                        Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
                    state->cov = (I_STATE - feature_vio_update_scale * feature_gain) * state->cov;
                  }
                  updateFrameState(*state);
                  feature_vio_commit_pass = true;
                  feature_vio_commit_status = 1;
                }
              }
              else
              {
                Eigen::Matrix<double, DIM_STATE, 1> state_update = Eigen::Matrix<double, DIM_STATE, 1>::Zero();
                state_update.block<6, 1>(0, 0) = scaled_solution;
                (*state) += state_update;
                if (feature_vio_inekf_update_en && update_covariance)
                {
                  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> I_STATE =
                      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
                  state->cov = (I_STATE - feature_vio_update_scale * feature_gain) * state->cov;
                }
                updateFrameState(*state);
                feature_vio_commit_pass = true;
                feature_vio_commit_status = 1;
              }
            }
          }
        }
      }
    }
  }

  feature_vio_solve_time = omp_get_wtime() - t0 - feature_vio_track_time;
  const double landmark_t0 = omp_get_wtime();
  std::vector<FeatureVioLandmark> next_landmarks;
  buildFeatureVioLandmarks(img, pg, next_landmarks);
  feature_vio_landmark_time = omp_get_wtime() - landmark_t0;
  feature_vio_prev_landmarks_.swap(next_landmarks);
  if (feature_vio_lk_pyramid_cache_en && feature_vio_lk_temporal_cache_en)
  {
    if (!temporal_pyramid_updated)
    {
      cv::buildOpticalFlowPyramid(img, cur_pyramid, lk_window_size, lk_max_level, false, cv::BORDER_REFLECT_101,
                                  cv::BORDER_CONSTANT, false);
      feature_vio_prev_lk_pyramid_.swap(cur_pyramid);
      feature_vio_prev_lk_window_ = lk_window;
      feature_vio_prev_lk_max_level_ = lk_max_level;
    }
  }
  else
  {
    feature_vio_prev_lk_pyramid_.clear();
    feature_vio_prev_lk_window_ = 0;
    feature_vio_prev_lk_max_level_ = -1;
  }
  feature_vio_prev_img_ = img.clone();
  feature_vio_diag_time = omp_get_wtime() - t0;

  printf("[FEATURE_VIO_STATS] frame=%d enabled=%d prev_landmarks=%zu tracked=%zu valid_reproj=%zu inliers=%zu "
         "inlier_ratio=%.6f reproj_rmse=%.6f mean_parallax=%.6f fb_check_run=%d fb_inliers=%zu fb_rejects=%zu "
         "ransac_inliers=%zu robust_inliers=%zu robust_inlier_ratio=%.6f robust_reproj_rmse=%.6f gate_pass=%d "
         "robust_depth_mean=%.6f robust_depth_min=%.6f robust_depth_max=%.6f robust_ref_depth_mean=%.6f "
         "robust_depth_px_dist_mean=%.6f robust_reproj_bias_u=%.6f robust_reproj_bias_v=%.6f robust_abs_reproj_mean=%.6f "
         "solve_points=%zu dry_run_status=%d dry_run_update_norm=%.9f dry_run_rot_deg=%.9f dry_run_pos_norm=%.9f "
         "dry_run_condition=%.6f dry_run_eig_min=%.9f dry_run_eig_max=%.9f "
         "update_en=%d inekf_update_en=%d img_point_cov=%.6f "
         "adaptive_depth_scene_mean=%.6f adaptive_inlier_scene_mean=%.6f "
         "adaptive_scene_samples=%zu adaptive_weight_en=%d "
         "adaptive_weight_mean=%.6f adaptive_weight_min=%.6f adaptive_effective_points=%.6f update_scale=%.6f "
         "feature_vio_saif_enabled=%d feature_vio_saif_gated_dirs=%zu feature_vio_saif_min_weight=%.6f feature_vio_saif_weight_avg=%.6f "
         "commit_status=%d commit_pass=%d commit_update_norm=%.9f commit_rot_deg=%.9f commit_pos_norm=%.9f "
         "commit_rot_x=%.9f commit_rot_y=%.9f commit_rot_z=%.9f commit_pos_x=%.9f commit_pos_y=%.9f commit_pos_z=%.9f "
         "lio_weak_dir=%d lio_weak_abs_projection=%.9f lio_weak_abs_cosine=%.9f "
         "bias_watchdog_en=%d bias_watchdog_samples=%zu bias_watchdog_rot_mean_norm=%.9f bias_watchdog_pos_mean_norm=%.9f "
         "bias_watchdog_reject=%d "
         "new_features=%zu depth_landmarks=%zu depth_ratio=%.6f track_time=%.6f solve_time=%.6f "
         "landmark_time=%.6f detect_time=%.6f depth_project_time=%.6f depth_match_time=%.6f diag_time=%.6f\n",
         frame_count, feature_vio_diagnostic_en ? 1 : 0, feature_vio_prev_landmarks, feature_vio_tracked, feature_vio_valid_reproj,
         feature_vio_inliers, feature_vio_inlier_ratio, feature_vio_reproj_rmse, feature_vio_mean_parallax,
         feature_vio_fb_check_run ? 1 : 0, feature_vio_fb_inliers,
         feature_vio_fb_rejects, feature_vio_ransac_inliers, feature_vio_robust_inliers, feature_vio_robust_inlier_ratio,
         feature_vio_robust_reproj_rmse, feature_vio_gate_pass ? 1 : 0, feature_vio_robust_depth_mean, feature_vio_robust_depth_min,
         feature_vio_robust_depth_max, feature_vio_robust_ref_depth_mean, feature_vio_robust_depth_px_dist_mean,
         feature_vio_robust_reproj_bias_u, feature_vio_robust_reproj_bias_v, feature_vio_robust_abs_reproj_mean,
         feature_vio_solve_points, feature_vio_dry_run_status, feature_vio_dry_run_update_norm,
         feature_vio_dry_run_rot_deg, feature_vio_dry_run_pos_norm, feature_vio_dry_run_condition, feature_vio_dry_run_eig_min,
         feature_vio_dry_run_eig_max, feature_vio_update_en ? 1 : 0, feature_vio_inekf_update_en ? 1 : 0,
         feature_vio_img_point_cov, feature_vio_adaptive_depth_scene_mean, feature_vio_adaptive_inlier_scene_mean,
         feature_vio_adaptive_depth_scene_samples_,
         feature_vio_adaptive_weight_en ? 1 : 0, feature_vio_weight_mean,
         feature_vio_weight_min, feature_vio_effective_points, feature_vio_update_scale, saif_gate_en ? 1 : 0,
         feature_vio_saif_gated_dirs,
         feature_vio_saif_min_weight, feature_vio_saif_weight_avg, feature_vio_commit_status,
         feature_vio_commit_pass ? 1 : 0, feature_vio_commit_update_norm, feature_vio_commit_rot_deg, feature_vio_commit_pos_norm,
         feature_vio_commit_rot_x, feature_vio_commit_rot_y, feature_vio_commit_rot_z, feature_vio_commit_pos_x, feature_vio_commit_pos_y,
         feature_vio_commit_pos_z, feature_vio_lio_weak_dir, feature_vio_lio_weak_abs_projection, feature_vio_lio_weak_abs_cosine,
         feature_vio_bias_watchdog_en ? 1 : 0, feature_vio_bias_watchdog_samples,
         feature_vio_bias_watchdog_rot_mean_norm, feature_vio_bias_watchdog_pos_mean_norm, feature_vio_bias_watchdog_reject ? 1 : 0,
         feature_vio_new_features, feature_vio_depth_landmarks,
         feature_vio_new_features > 0 ? static_cast<double>(feature_vio_depth_landmarks) / static_cast<double>(feature_vio_new_features) : 0.0,
         feature_vio_track_time, feature_vio_solve_time, feature_vio_landmark_time, feature_vio_detect_time,
         feature_vio_depth_project_time, feature_vio_depth_match_time, feature_vio_diag_time);
}

void VIOManager::processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time)
{
  if (width != img.cols || height != img.rows)
  {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img, cv::Size(img.cols * image_resize_factor, img.rows * image_resize_factor), 0, 0, CV_INTER_LINEAR);
  }
  img_rgb = img.clone();
  img_cp = img.clone();
  // img_test = img.clone();

  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);

  new_frame_.reset(new Frame(cam, img));
  updateFrameState(*state);
  
  resetGrid();
  compute_jacobian_time = update_ekf_time = 0.0;
  saif_gated_dirs = 0;
  saif_frame_min_weight = 1.0;
  saif_frame_weight_avg = 1.0;

  const int feature_stride = std::max(1, feature_vio_frame_stride);
  const bool feature_vio_run = feature_vio_diagnostic_en && frame_count % feature_stride == 0;
  if (feature_vio_run)
  {
    runFeatureVioDiagnostic(img, pg);
  }
  else
  {
    feature_vio_diag_time = 0.0;
  }

  double t1 = omp_get_wtime();

  if (photometric_update_en) retrieveFromVisualSparseMap(img, pg, feat_map);

  double t2 = omp_get_wtime();

  if (photometric_update_en) computeJacobianAndUpdateEKF(img);

  double t3 = omp_get_wtime();

  if (photometric_update_en) generateVisualMapPoints(img, pg);

  double t4 = omp_get_wtime();
  
  if (photometric_update_en) plotTrackedPoints();

  if (photometric_update_en && plot_flag) projectPatchFromRefToCur(feat_map);

  double t5 = omp_get_wtime();

  if (photometric_update_en) updateVisualMapPoints(img);

  double t6 = omp_get_wtime();

  if (photometric_update_en) updateReferencePatch(feat_map);

  double t7 = omp_get_wtime();
  
  if(photometric_update_en && colmap_output_en)  dumpDataForColmap();

  frame_count++;
  ave_total = ave_total * (frame_count - 1) / frame_count + (t7 - t1 - (t5 - t4)) / frame_count;

  // printf("[ VIO ] feat_map.size(): %zu\n", feat_map.size());
  // printf("\033[1;32m[ VIO time ]: current frame: retrieveFromVisualSparseMap time: %.6lf secs.\033[0m\n", t2 - t1);
  // printf("\033[1;32m[ VIO time ]: current frame: computeJacobianAndUpdateEKF time: %.6lf secs, comp H: %.6lf secs, ekf: %.6lf secs.\033[0m\n", t3 - t2, computeH, ekf_time);
  // printf("\033[1;32m[ VIO time ]: current frame: generateVisualMapPoints time: %.6lf secs.\033[0m\n", t4 - t3);
  // printf("\033[1;32m[ VIO time ]: current frame: updateVisualMapPoints time: %.6lf secs.\033[0m\n", t6 - t5);
  // printf("\033[1;32m[ VIO time ]: current frame: updateReferencePatch time: %.6lf secs.\033[0m\n", t7 - t6);
  // printf("\033[1;32m[ VIO time ]: current total time: %.6lf, average total time: %.6lf secs.\033[0m\n", t7 - t1 - (t5 - t4), ave_total);

  // ave_build_residual_time = ave_build_residual_time * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;
  // ave_ekf_time = ave_ekf_time * (frame_count - 1) / frame_count + (t3 - t2) / frame_count;
 
  // cout << BLUE << "ave_build_residual_time: " << ave_build_residual_time << RESET << endl;
  // cout << BLUE << "ave_ekf_time: " << ave_ekf_time << RESET << endl;
  
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         VIO Time                            |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t2 - t1);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("[VIO_STATS] frame=%d sparse_map_size=%zu total_points=%d inverse_composition=%d photometric_update=%d "
         "feature_vio_diag=%d feature_vio_run=%d feature_vio_stride=%d feature_vio_time=%.6f ref_patch_cache=%d "
         "vio_saif_enabled=%d vio_saif_gated_dirs=%zu vio_saif_min_weight=%.6f vio_saif_weight_avg=%.6f "
         "retrieve_time=%.6f compute_update_time=%.6f compute_jacobian_time=%.6f update_ekf_time=%.6f "
         "generate_points_time=%.6f plot_time=%.6f update_points_time=%.6f update_ref_patch_time=%.6f "
         "total_time=%.6f total_with_feature_time=%.6f avg_total_time=%.6f "
         "vio_map_pg_size=%zu vio_map_normal_zero=%zu vio_map_in_frame=%zu vio_map_grid_map_skip=%zu "
         "vio_map_selected_scan=%zu vio_map_voxel_candidates=%zu vio_map_selected_voxel=%zu vio_map_added=%zu "
         "vio_map_depth_positive=%zu vio_map_shi_max=%.3f "
         "vio_map_u_min=%.3f vio_map_u_max=%.3f vio_map_v_min=%.3f vio_map_v_max=%.3f vio_map_z_min=%.3f vio_map_z_max=%.3f\n",
         frame_count, feat_map.size(), total_points, inverse_composition_en ? 1 : 0, photometric_update_en ? 1 : 0,
         feature_vio_diagnostic_en ? 1 : 0, feature_vio_run ? 1 : 0, feature_stride, feature_vio_diag_time,
         has_ref_patch_cache ? 1 : 0, saif_gate_en ? 1 : 0,
         saif_gated_dirs, saif_frame_min_weight, saif_frame_weight_avg, t2 - t1, t3 - t2,
         compute_jacobian_time, update_ekf_time, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t7 - t1 - (t5 - t4),
         feature_vio_diag_time + t7 - t1 - (t5 - t4), ave_total,
         vio_map_pg_size, vio_map_normal_zero, vio_map_in_frame, vio_map_grid_map_skip, vio_map_selected_from_scan, vio_map_voxel_candidates,
         vio_map_selected_from_voxel, vio_map_added, vio_map_depth_positive, vio_map_shi_max, vio_map_u_min, vio_map_u_max, vio_map_v_min,
         vio_map_v_max, vio_map_z_min, vio_map_z_max);

  // std::string text = std::to_string(int(1 / (t7 - t1 - (t5 - t4)))) + " HZ";
  // cv::Point2f origin;
  // origin.x = 20;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
  // cv::imwrite("/home/chunran/Desktop/raycasting/" + std::to_string(new_frame_->id_) + ".png", img_cp);
}
