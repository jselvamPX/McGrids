#include "fast_mcmt.hpp"
#include "kdtree.hpp"
#include <tbb/tbb.h>

namespace GEO
{
    MCMT::MCMT()
    {
        GEO::initialize();
        delaunay_ = new PeriodicDelaunay3d(periodic_, 1.0);
        if (!periodic_)
        {
            delaunay_->set_keeps_infinite(true);
        }
    }

    MCMT::~MCMT()
    {
        delete delaunay_;
    }

    void MCMT::clear()
    {
        delete delaunay_;
        num_point_visited_ = 0;
        point_positions_.clear();
        point_values_.clear();
        point_errors_.clear();

        // create new delaunay
        delaunay_ = new PeriodicDelaunay3d(periodic_, 1.0);
        if (!periodic_)
        {
            delaunay_->set_keeps_infinite(true);
        }
    }

    std::vector<double> MCMT::get_grid_points()
    {
        std::vector<double> grid_points;
        for (index_t v = 0; v < delaunay_->nb_vertices(); v++)
        {
            grid_points.push_back(delaunay_->vertex_ptr(v)[0]);
            grid_points.push_back(delaunay_->vertex_ptr(v)[1]);
            grid_points.push_back(delaunay_->vertex_ptr(v)[2]);
        }
        return grid_points;
    }

    std::vector<int> MCMT::get_grids()
    {
        std::vector<int> v_indices;

        for (int i = 0; i < delaunay_->nb_finite_cells(); i++)
        {
            for (index_t lv = 0; lv < 4; ++lv)
            {
                int v = delaunay_->cell_vertex(i, lv);
                double x = delaunay_->vertex_ptr(v)[0];
                v_indices.push_back(int(v));
            }
        }
        return v_indices;
    }

    void MCMT::add_points(int num_points, double *point_positions, double *point_values)
    {
        num_point_visited_ = 0;
        int current_num_points = point_positions_.size() / 3;
        for (index_t i = 0; i < num_points; i++)
        {
            point_positions_.push_back(point_positions[i * 3]);
            point_positions_.push_back(point_positions[i * 3 + 1]);
            point_positions_.push_back(point_positions[i * 3 + 2]);
            point_values_.push_back(point_values[i]);
            point_errors_.push_back(1 / (abs(point_values[i]) + 1e-6));
        }
        delete delaunay_;
        delaunay_ = new PeriodicDelaunay3d(periodic_, 1.0);
        if (!periodic_)
        {
            delaunay_->set_keeps_infinite(true);
        }
        delaunay_->set_vertices(point_positions_.size() / 3, point_positions_.data());
        delaunay_->compute();

        for (int i = 0; i < point_positions_.size(); i++)
        {
            if (point_positions_[i] > max_bound)
            {
                max_bound = point_positions_[i];
            }
            if (point_positions_[i] < min_bound)
            {
                min_bound = point_positions_[i];
            }
        }

        for (int i = current_num_points; i < point_positions_.size() / 3; i++)
        {
            ConvexCell C;
            PeriodicDelaunay3d::IncidentTetrahedra W;

            get_cell(i, C, W);
            double volume = C.volume();
            point_volumes_.push_back(volume);
            volume_changed_.push_back(false);
        }
        for (int i = current_num_points; i < point_positions_.size() / 3; i++)
        {
            PeriodicDelaunay3d::IncidentTetrahedra W;
            delaunay_->get_incident_tets(i, W);
            for (auto it = W.begin(); it != W.end(); it++)
            {
                for (int lv = 0; lv < 4; lv++)
                {
                    int point_id = delaunay_->cell_vertex(*it, lv);
                    if (point_id == -1 || point_id == i)
                        continue;
                    volume_changed_[point_id] = true;
                }
            }
        }

        for (int i = 0; i < point_positions_.size() / 3; i++)
        {
            if (volume_changed_[i])
            {
                ConvexCell C;
                PeriodicDelaunay3d::IncidentTetrahedra W;

                get_cell(i, C, W);
                point_volumes_[i] = C.volume();
            }
        }
        // exit(0);
    }

    void MCMT::add_mid_points(int num_points, double* point_positions, double* point_values)
    {
        auto start = std::chrono::high_resolution_clock::now();
        num_point_visited_ = point_positions_.size() / 3;
        size_t old_positions_size = point_positions_.size();
        size_t old_values_size = point_values_.size();

        // Preallocate memory
        point_positions_.resize(old_positions_size + num_points * 3);
        point_values_.resize(old_values_size + num_points);
        point_errors_.resize(old_values_size + num_points);

        double max_b = max_bound;
        double min_b = min_bound;

        for (index_t i = 0; i < num_points; ++i)
        {
            double x = point_positions[i * 3];
            double y = point_positions[i * 3 + 1];
            double z = point_positions[i * 3 + 2];

            // Direct indexing
            size_t idx = old_positions_size + i * 3;
            point_positions_[idx] = x;
            point_positions_[idx + 1] = y;
            point_positions_[idx + 2] = z;

            point_values_[old_values_size + i] = point_values[i];
            point_errors_[old_values_size + i] = 1 / (std::abs(point_values[i]) + 1e-6);

            // Optimize bound updates
            if (x > max_b) max_b = x;
            if (y > max_b) max_b = y;
            if (z > max_b) max_b = z;

            if (x < min_b) min_b = x;
            if (y < min_b) min_b = y;
            if (z < min_b) min_b = z;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        diff = diff * 1000;
        std::cout << "add_mid_points: Loop Time: " << diff.count() << " ms\n";

        // Update bounds
        max_bound = max_b;
        min_bound = min_b;

        delete delaunay_;
        delaunay_ = new PeriodicDelaunay3d(periodic_, 1.0);

        if (!periodic_)
        {
            delaunay_->set_keeps_infinite(true);
        }

        // Set initial vertices
        delaunay_->set_vertices(point_positions_.size() / 3, point_positions_.data());
        // Compute the updated triangulation
        start = std::chrono::high_resolution_clock::now();
        delaunay_->compute();

        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        diff = diff * 1000;
        std::cout << "add_mid_points: delaunay computation time: " << diff.count() << " ms\n";
    }


    std::vector<double> MCMT::interpolate(double *point1, double *point2, double sd1, double sd2)
    {
        double p1_x = point1[0];
        double p1_y = point1[1];
        double p1_z = point1[2];
        double p2_x = point2[0];
        double p2_y = point2[1];
        double p2_z = point2[2];
        double t = sd1 / ((sd1 - sd2));
        if (abs(sd1 - sd2) < 1e-6)
        {
            // std::cout << "WARNING! SD1 == SD2" << std::endl;
            t = 0.5;
        }
        return std::vector<double>{p1_x + t * (p2_x - p1_x), p1_y + t * (p2_y - p1_y), p1_z + t * (p2_z - p1_z)};
    }

    std::vector<double> MCMT::sample_points_rejection(int num_points, double min_bound, double max_bound)
    {
        // compute density
        KDTree tree = KDTree(point_positions_.size() / 3, point_positions_.data(), point_errors_.data());
        int current_num_points = 0;
        int batch_size = 4096;
        std::vector<double> sampled_points;
        while (current_num_points < num_points)
        {
            std::vector<double> new_points;
            new_points.reserve(batch_size * 3);
            for (int i = 0; i < batch_size * 3; i++)
            {
                new_points.push_back(Numeric::random_float64() * (max_bound - min_bound) + min_bound);
            }
            std::vector<double> density = tree.compute_density(batch_size, new_points.data());
            double max_density = *std::max_element(density.begin(), density.end());
            for (int i = 0; i < batch_size; i++)
            {
                double threshold = Numeric::random_float64() * max_density;
                if (density[i] > threshold)
                {
                    sampled_points.push_back(new_points[i * 3]);
                    sampled_points.push_back(new_points[i * 3 + 1]);
                    sampled_points.push_back(new_points[i * 3 + 2]);
                    current_num_points++;
                    if (current_num_points == num_points)
                        break;
                }
            }
        }
        return sampled_points;
    }

    double MCMT::tetrahedronVolume(const std::vector<double> &coordinates)
    {
        // Check if there are enough coordinates
        if (coordinates.size() != 12)
        {
            std::cerr << "Error: Insufficient coordinates provided." << std::endl;
            return 0.0; // Return an appropriate value
        }

        // Extract coordinates of points A, B, C, and D
        double Ax = coordinates[0], Ay = coordinates[1], Az = coordinates[2];
        double Bx = coordinates[3], By = coordinates[4], Bz = coordinates[5];
        double Cx = coordinates[6], Cy = coordinates[7], Cz = coordinates[8];
        double Dx = coordinates[9], Dy = coordinates[10], Dz = coordinates[11];

        // Vector from A to B
        double ABx = Bx - Ax, ABy = By - Ay, ABz = Bz - Az;

        // Vector from A to C
        double ACx = Cx - Ax, ACy = Cy - Ay, ACz = Cz - Az;

        // Vector from A to D
        double ADx = Dx - Ax, ADy = Dy - Ay, ADz = Dz - Az;

        // Cross product of AC and AD
        double crossProduct_i = ACy * ADz - ACz * ADy;
        double crossProduct_j = ACz * ADx - ACx * ADz;
        double crossProduct_k = ACx * ADy - ACy * ADx;

        // Dot product of AB and the cross product of AC and AD
        double dotProduct = ABx * crossProduct_i + ABy * crossProduct_j + ABz * crossProduct_k;

        // Volume calculation
        double volume = std::abs(dotProduct) / 6.0;

        return volume;
    }

    std::vector<double> MCMT::compute_voronoi_error()
    {

        std::vector<double> voronoi_errors;

        for (int i = 0; i < delaunay_->nb_vertices(); i++)
        {
            double volume = point_volumes_[i];
            voronoi_errors.push_back(point_errors_[i] * volume);
        }

        return voronoi_errors;
    }

    std::vector<double> MCMT::compute_tet_error()
    {
        std::vector<double> tet_errors;
        tet_errors.resize(delaunay_->nb_finite_cells());

        tbb::parallel_for(tbb::blocked_range<int>(0, delaunay_->nb_finite_cells()),
                          [&](tbb::blocked_range<int> ti)
                          {
                              for (int i = ti.begin(); i < ti.end(); i++)
                              {
                                  double tet_density = 0;
                                  std::vector<double> point_coordinates;

                                  for (index_t lv = 0; lv < 4; ++lv)
                                  {
                                      int v = delaunay_->cell_vertex(i, lv);
                                      tet_density += point_errors_[v];
                                      for (int c = 0; c < 3; c++)
                                      {
                                          point_coordinates.push_back(point_positions_[v * 3 + c]);
                                      }
                                  }
                                  tet_errors[i] = tet_density * tetrahedronVolume(point_coordinates);
                              }
                          });
        return tet_errors;
    }

    int findBounds(const std::vector<double> &vec, float x)
    {
        int low = 0;
        int high = vec.size() - 1;

        while (low < high)
        {
            int mid = (low + high) / 2;

            if (vec[mid] < x)
            {
                low = mid + 1;
            }
            else
            {
                high = mid;
            }
        }

        if (low > 0 && std::abs(vec[low - 1] - x) < std::abs(vec[low] - x))
        {
            return low - 1;
        }
        else
        {
            return low;
        }
    }

    std::vector<double> MCMT::sample_polytope(int vor_index)
    {
        // vor_index = 10;
        ConvexCell C;
        PeriodicDelaunay3d::IncidentTetrahedra W;
        get_cell(vor_index, C, W);
        vec3 g;
        g = C.barycenter();
        int index = 1;
        // For each vertex of the Voronoi cell
        // Start at 1 (vertex 0 is point at infinity)
        std::vector<std::vector<double>> tetrahedrons;

        for (index_t v = 1; v < C.nb_v(); ++v)
        {
            index_t t = C.vertex_triangle(v);
            //  Happens if a clipping plane did not
            // clip anything.
            if (t == VBW::END_OF_LIST)
            {
                continue;
            }

            //   Now iterate on the Voronoi vertices of the
            // Voronoi facet. In dual form this means iterating
            // on the triangles incident to vertex v
            vec3 P[3];
            index_t n = 0;

            do
            {
                std::vector<double> tetrahedron;

                //   Triangulate the Voronoi facet and send the
                // triangles to OpenGL/GLUP.
                if (n == 0)
                {
                    P[0] = C.triangle_point(VBW::ushort(t));
                }
                else if (n == 1)
                {
                    P[1] = C.triangle_point(VBW::ushort(t));
                }
                else
                {
                    P[2] = C.triangle_point(VBW::ushort(t));

                    for (index_t i = 0; i < 3; ++i)
                    {
                        tetrahedron.push_back(P[i].x);
                        tetrahedron.push_back(P[i].y);
                        tetrahedron.push_back(P[i].z);
                        // output_mesh << "v " << P[i].x << " " << P[i].y << " " << P[i].z << std::endl;
                    }
                    tetrahedron.push_back(g.x);
                    tetrahedron.push_back(g.y);
                    tetrahedron.push_back(g.z);
                    tetrahedrons.push_back(tetrahedron);
                    // output_mesh << "f " << index << "  " << index + 1 << " " << index + 2 << std::endl;
                    index += 3;
                    P[1] = P[2];
                }
                index_t lv = C.triangle_find_vertex(t, v);
                t = C.triangle_adjacent(t, (lv + 1) % 3);
                ++n;
            } while (t != C.vertex_triangle(v));
        }
        std::vector<double> tetrahedron_volumes;
        for (int i = 0; i < tetrahedrons.size(); i++)
        {
            tetrahedron_volumes.push_back(tetrahedronVolume(tetrahedrons[i]));
        }

        double tet_volume_sum = std::accumulate(tetrahedron_volumes.begin(), tetrahedron_volumes.end(), 0.0);

        for (int i = 0; i < tetrahedron_volumes.size(); i++)
        {
            tetrahedron_volumes[i] /= tet_volume_sum;
        }

        std::vector<double> cumsum;
        double current_sum = 0;
        for (int i = 0; i < tetrahedron_volumes.size(); i++)
        {
            current_sum += tetrahedron_volumes[i];
            cumsum.push_back(current_sum);
        }

        auto upper = std::upper_bound(cumsum.begin(), cumsum.end(), Numeric::random_float64());
        int tet_index = std::distance(cumsum.begin(), upper);
        return sample_tet(tetrahedrons[tet_index]);
    }

    std::vector<double> MCMT::sample_points_voronoi(const int num_points)
    {
        std::vector<double> voronoi_density = compute_voronoi_error();

        double voronoi_density_sum = std::accumulate(voronoi_density.begin(), voronoi_density.end(), 0.0);

        tbb::parallel_for(tbb::blocked_range<int>(0, voronoi_density.size()),
                          [&](tbb::blocked_range<int> ti)
                          {
                              for (int i = ti.begin(); i < ti.end(); i++)
                              {
                                  voronoi_density[i] /= voronoi_density_sum;
                              }
                          });

        std::vector<double> cumsum;
        double current_sum = 0;
        for (int i = 0; i < voronoi_density.size(); i++)
        {
            current_sum += voronoi_density[i];
            cumsum.push_back(current_sum);
        }
        tbb::concurrent_vector<std::vector<double>> sample_points;
        tbb::parallel_for(tbb::blocked_range<int>(0, num_points),
                          [&](tbb::blocked_range<int> ti)
                          {
                              for (int i = ti.begin(); i < ti.end(); i++)
                              {
                                  auto upper = std::upper_bound(cumsum.begin(), cumsum.end(), Numeric::random_float64());
                                  int voro_index = std::distance(cumsum.begin(), upper);
                                  std::vector<double> sampled_point = sample_polytope(voro_index);
                                  // sample_points.push_back(sampled_point[0]);
                                  // sample_points.push_back(sampled_point[1]);
                                  // // sample_points.push_back(sampled_point[2]);
                                  // sample_points[i * 3] = sampled_point[0];
                                  // sample_points[i * 3 + 1] = sampled_point[1];
                                  // sample_points[i * 3 + 2] = sampled_point[2];
                                  sample_points.push_back(std::vector<double>{sampled_point[0], sampled_point[1], sampled_point[2]});
                              }
                          });
        // std::vector<double> sample_points_vec(sample_points.begin(), sample_points.end());
        std::vector<double> sample_points_vec;
        for (int i = 0; i < sample_points.size(); i++)
        {
            sample_points_vec.push_back(sample_points[i][0]);
            sample_points_vec.push_back(sample_points[i][1]);
            sample_points_vec.push_back(sample_points[i][2]);
        }
        return sample_points_vec;
    }

    std::vector<double> MCMT::sample_tet(std::vector<double> point_positions)
    {
        double s = Numeric::random_float64();
        double t = Numeric::random_float64();
        double u = Numeric::random_float64();
        if (s + t > 1.0)
        {
            s = 1.0 - s;
            t = 1.0 - t;
        }
        if (t + u > 1.0)
        {
            double tmp = u;
            u = 1.0 - s - t;
            t = 1.0 - tmp;
        }
        else if (s + t + u > 1.0)
        {
            double tmp = u;
            u = s + t + u - 1.0;
            s = 1 - t - tmp;
        }
        double a = 1 - s - t - u;

        double x = a * point_positions[0] + s * point_positions[3] + t * point_positions[6] + u * point_positions[9];
        double y = a * point_positions[1] + s * point_positions[4] + t * point_positions[7] + u * point_positions[10];
        double z = a * point_positions[2] + s * point_positions[5] + t * point_positions[8] + u * point_positions[11];
        return std::vector<double>{x, y, z};
    }

    std::vector<double> MCMT::compute_face_mid_point(int num_points, const std::vector<double> &points)
    {
        double mid_point_x = 0;
        double mid_point_y = 0;
        double mid_point_z = 0;

        for (int i = 0; i < num_points; i++)
        {
            mid_point_x += points[i * 3];
            mid_point_y += points[i * 3 + 1];
            mid_point_z += points[i * 3 + 2];
        }
        mid_point_x /= num_points;
        mid_point_y /= num_points;
        mid_point_z /= num_points;
        return std::vector<double>{mid_point_x, mid_point_y, mid_point_z};
    }

    std::vector<double> MCMT::get_mid_points()
    {
        tbb::concurrent_set<size_t> new_cells;
        // if there is a bug, roll back...
        if (delaunay_->nb_finite_cells() == 0)
        {
            point_positions_ = std::vector<double>(point_positions_.begin(), point_positions_.begin() + num_point_visited_ * 3);
            point_values_ = std::vector<double>(point_values_.begin(), point_values_.begin() + num_point_visited_);
            point_errors_ = std::vector<double>(point_errors_.begin(), point_errors_.begin() + num_point_visited_);

            delete delaunay_;
            delaunay_ = new PeriodicDelaunay3d(periodic_, 1.0);
            if (!periodic_)
            {
                delaunay_->set_keeps_infinite(true);
            }
            delaunay_->set_vertices(point_positions_.size() / 3, point_positions_.data());
            delaunay_->compute();
            return std::vector<double>{};
        }

        tbb::parallel_for(tbb::blocked_range<int>(num_point_visited_, point_positions_.size() / 3),
                          [&](tbb::blocked_range<int> ti)
                          {
                              for (int i = ti.begin(); i < ti.end(); i++)
                              {
                                  PeriodicDelaunay3d::IncidentTetrahedra W;
                                  delaunay_->get_incident_tets(i, W);
                                  for (auto it = W.begin(); it != W.end(); it++)
                                  {
                                      if (*it < delaunay_->nb_finite_cells() && *it >= 0)
                                      {
                                          bool skip = false;
                                          for (int lv = 0; lv < 4; lv++)
                                          {
                                              int v = delaunay_->cell_vertex(*it, lv);
                                              if (v == -1)
                                              {
                                                  skip = true;
                                              }
                                              if (point_positions_[v * 3] - min_bound < 1e-6 || max_bound - point_positions_[v * 3] < 1e-6)
                                              {
                                                  skip = true;
                                              }
                                              if (point_positions_[v * 3 + 1] - min_bound < 1e-6 || max_bound - point_positions_[v * 3 + 1] < 1e-6)
                                              {
                                                  skip = true;
                                              }
                                              if (point_positions_[v * 3 + 2] - min_bound < 1e-6 || max_bound - point_positions_[v * 3 + 2] < 1e-6)
                                              {
                                                  skip = true;
                                              }
                                          }
                                          if (!skip)
                                          {
                                              new_cells.insert(*it);
                                          }
                                      }
                                  }
                              }
                          });

        std::vector<int> new_cell_ids(new_cells.begin(), new_cells.end());
        tbb::concurrent_vector<std::vector<double>> sample_points;
        tbb::parallel_for(tbb::blocked_range<int>(0, new_cell_ids.size()),
                          [&](tbb::blocked_range<int> ti)
                          {
						  for(index_t i = ti.begin(); i < ti.end(); i++)
						  {
                            index_t t = new_cell_ids[i];
                            unsigned char index = 0;

							  std::vector<int> tri2v;
							  for (index_t lv = 0; lv < 4; ++lv)
							  {
								  int v = delaunay_->cell_vertex(t, lv);
								  if (v != -1){
									  tri2v.push_back(int(v));
                                  }
                                if(point_values_[v] < 0){
                                    index |= (1 << lv);
                                }
							  }
                            if(index == 0x00 || index == 0x0F){
                                continue;
                            }

							  std::vector<double> intersection_points;
							  if (point_values_[tri2v[0]] * point_values_[tri2v[1]] < 0)
							  {
                                  std::vector<double> interpolated_point = interpolate(point_positions_.data() + tri2v[0] * 3, point_positions_.data() + tri2v[1] * 3, point_values_[tri2v[0]], point_values_[tri2v[1]]);
								  intersection_points.insert(intersection_points.end(), interpolated_point.begin(), interpolated_point.end());
							  }
							  if (point_values_[tri2v[0]] * point_values_[tri2v[2]] < 0)
							  {   std::vector<double> interpolated_point = interpolate(point_positions_.data() + tri2v[0] * 3, point_positions_.data() + tri2v[2] * 3, point_values_[tri2v[0]], point_values_[tri2v[2]]);
								  intersection_points.insert(intersection_points.end(), interpolated_point.begin(), interpolated_point.end());
							  }
							  if (point_values_[tri2v[0]] * point_values_[tri2v[3]] < 0)
							  {   std::vector<double> interpolated_point = interpolate(point_positions_.data() + tri2v[0] * 3, point_positions_.data() + tri2v[3] * 3, point_values_[tri2v[0]], point_values_[tri2v[3]]);
								  intersection_points.insert(intersection_points.end(), interpolated_point.begin(), interpolated_point.end());
							  }
							  if (point_values_[tri2v[1]] * point_values_[tri2v[2]] < 0)
							  {   std::vector<double> interpolated_point = interpolate(point_positions_.data() + tri2v[1] * 3, point_positions_.data() + tri2v[2] * 3, point_values_[tri2v[1]], point_values_[tri2v[2]]);
								  intersection_points.insert(intersection_points.end(), interpolated_point.begin(), interpolated_point.end());
							  }
							  if (point_values_[tri2v[1]] * point_values_[tri2v[3]] < 0)
							  {   std::vector<double> interpolated_point = interpolate(point_positions_.data() + tri2v[1] * 3, point_positions_.data() + tri2v[3] * 3, point_values_[tri2v[1]], point_values_[tri2v[3]]);
								  intersection_points.insert(intersection_points.end(), interpolated_point.begin(), interpolated_point.end());
							  }
							  if (point_values_[tri2v[2]] * point_values_[tri2v[3]] < 0)
							  {   std::vector<double> interpolated_point = interpolate(point_positions_.data() + tri2v[2] * 3, point_positions_.data() + tri2v[3] * 3, point_values_[tri2v[2]], point_values_[tri2v[3]]);
								  intersection_points.insert(intersection_points.end(), interpolated_point.begin(), interpolated_point.end());
							  }
							  if (intersection_points.size() != 0)
							  { 
								std::vector<double> mid_point = compute_face_mid_point(intersection_points.size()/3, intersection_points);

                                    bool skip_mid_point=false;
                                    std::vector<double> point_coordinates;
                                    double threshold = 1e-12;


                                    std::vector<std::vector<int>> tet_indices{{0,1,2},{0,2,3},{0,1,3},{1,2,3}};

                                    for(auto tet_index: tet_indices){

                                        point_coordinates.clear();
                                        for (int i:tet_index){
                                            point_coordinates.push_back(point_positions_[tri2v[i]*3]);
                                            point_coordinates.push_back(point_positions_[tri2v[i]*3+1]);
                                            point_coordinates.push_back(point_positions_[tri2v[i]*3+2]);

                                        }
                                        point_coordinates.push_back(mid_point[0]);
                                        point_coordinates.push_back(mid_point[1]);
                                        point_coordinates.push_back(mid_point[2]);

                                        double volume = tetrahedronVolume(point_coordinates);
                                        if (volume < threshold){
                                            skip_mid_point = true;
                                        }

                                    }

                                    if(! skip_mid_point){
                                        sample_points.push_back(mid_point);
                                    }
							
							  }
						  } });

        std::vector<double> new_points;
        new_points.resize(sample_points.size() * 3);
        for (int i = 0; i < sample_points.size(); i++)
        {
            new_points[i*3] = sample_points[i][0];
            new_points[i*3+1] = sample_points[i][1];
            new_points[i*3+2] = sample_points[i][2];

        }
        return new_points;
    }

    void MCMT::get_cell(index_t v, ConvexCell &C, PeriodicDelaunay3d::IncidentTetrahedra &W)
    {
        delaunay_->copy_Laguerre_cell_from_Delaunay(v, C, W);
        if (!periodic_)
        {
            C.clip_by_plane(vec4(1.0, 0.0, 0.0, max_bound));
            C.clip_by_plane(vec4(-1.0, 0.0, 0.0, -min_bound));
            C.clip_by_plane(vec4(0.0, 1.0, 0.0, max_bound));
            C.clip_by_plane(vec4(0.0, -1.0, 0.0, -min_bound));
            C.clip_by_plane(vec4(0.0, 0.0, 1.0, max_bound));
            C.clip_by_plane(vec4(0.0, 0.0, -1.0, -min_bound));
        }
        C.compute_geometry();
    }
    std::vector<double> MCMT::lloyd_relaxation(double *relaxed_point_positions, int num_points, int num_iter)
    {
        int current_num_points = point_positions_.size() / 3;

        // Combine existing and new points into all_points
        std::vector<double> all_points(point_positions_.size() + num_points * 3);
        std::copy(point_positions_.begin(), point_positions_.end(), all_points.begin());
        std::copy(relaxed_point_positions, relaxed_point_positions + num_points * 3, all_points.begin() + point_positions_.size());

        // Initialize delaunay_ if necessary
        if (!delaunay_)
        {
            delaunay_ = new PeriodicDelaunay3d(periodic_, 1.0);
            if (!periodic_)
                delaunay_->set_keeps_infinite(true);
        }

        // Set vertices and compute Delaunay triangulation
        delaunay_->set_vertices(all_points.size() / 3, all_points.data());
        auto start = std::chrono::high_resolution_clock::now();
        delaunay_->compute();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        diff = diff * 1000;
        std::cout << "lloyd relaxation: delaunay compute Time: " << diff.count() << " ms\n";


        // Preallocate updated_lloyd_points
        std::vector<double> updated_lloyd_points(all_points.size());

        double range = max_bound - min_bound;

        for (int iter = 0; iter < num_iter; iter++)
        {
            // Copy existing points
            std::copy(all_points.begin(), all_points.begin() + current_num_points * 3, updated_lloyd_points.begin());

            // Parallelize the computation of new points
            tbb::parallel_for(tbb::blocked_range<index_t>(current_num_points, all_points.size() / 3),
                [&](const tbb::blocked_range<index_t>& r) {
                    for (index_t v = r.begin(); v != r.end(); ++v)
                    {
                        PeriodicDelaunay3d::IncidentTetrahedra W;
                        ConvexCell C;
                        get_cell(v, C, W);
                        vec3 g = C.barycenter();

                        updated_lloyd_points[3 * v] = g.x;
                        updated_lloyd_points[3 * v + 1] = g.y;
                        updated_lloyd_points[3 * v + 2] = g.z;

                        // Adjust for periodicity
                        for (int d = 0; d < 3; ++d)
                        {
                            double& coord = updated_lloyd_points[3 * v + d];
                            coord = min_bound + std::fmod(coord - min_bound + range, range);
                        }
                    }
                });

            // Swap all_points and updated_lloyd_points
            std::swap(all_points, updated_lloyd_points);

            // Update delaunay_ with new positions
            delaunay_->set_vertices(all_points.size() / 3, all_points.data());
            start = std::chrono::high_resolution_clock::now();
            delaunay_->compute();
            end = std::chrono::high_resolution_clock::now();
            diff = end - start;
            diff = diff * 1000;
            std::cout << "lloyd relaxation: delaunay new compute Time: " << diff.count() << " ms\n";
        }

        // Extract the relaxed points
        std::vector<double> points_vec(all_points.begin() + current_num_points * 3, all_points.end());

        return points_vec;
    }

    void MCMT::output_grid_points(std::string filename)
    {
        std::ofstream outfile(filename);
        for (int i = 0; i < point_positions_.size() / 3; i++)
        {
            outfile << "v " << point_positions_[i * 3] << " " << point_positions_[i * 3 + 1] << " " << point_positions_[i * 3 + 2] << std::endl;
        }
    }

    void MCMT::save_triangle_mesh(std::string filename)
    {

        std::vector<std::vector<double>> mesh_vertices;
        std::vector<std::vector<int>> mesh_faces;

        std::map<std::pair<int, int>, int> ev_map;
        int v_counter = 0;
        std::vector<int> intersection_cell;

        for (int i = 0; i < delaunay_->nb_finite_cells(); i++)
        {
            std::vector<int> v_indices;
            bool flag = false;
            for (index_t lv = 0; lv < 4; ++lv)
            {
                int v = delaunay_->cell_vertex(i, lv);
                v_indices.push_back(int(v));
            }

            if (point_values_[v_indices[0]] * point_values_[v_indices[1]] < 0)
            {
                flag = true;
                std::pair<int, int> e01 = (v_indices[0] < v_indices[1]) ? std::make_pair(v_indices[0], v_indices[1]) : std::make_pair(v_indices[1], v_indices[0]);
                if (ev_map.find(e01) == ev_map.end())
                {
                    std::vector<double> interpolated_point = interpolate(point_positions_.data() + v_indices[0] * 3, point_positions_.data() + v_indices[1] * 3, point_values_[v_indices[0]], point_values_[v_indices[1]]);
                    ev_map.insert({e01, mesh_vertices.size()});
                    mesh_vertices.push_back(interpolated_point);
                }
            }
            if (point_values_[v_indices[0]] * point_values_[v_indices[2]] < 0)
            {
                flag = true;
                std::pair<int, int> e02 = (v_indices[0] < v_indices[2]) ? std::make_pair(v_indices[0], v_indices[2]) : std::make_pair(v_indices[2], v_indices[0]);
                if (ev_map.find(e02) == ev_map.end())
                {
                    std::vector<double> interpolated_point = interpolate(point_positions_.data() + v_indices[0] * 3, point_positions_.data() + v_indices[2] * 3, point_values_[v_indices[0]], point_values_[v_indices[2]]);
                    ev_map.insert({e02, mesh_vertices.size()});
                    mesh_vertices.push_back(interpolated_point);
                }
            }
            if (point_values_[v_indices[0]] * point_values_[v_indices[3]] < 0)
            {
                flag = true;
                std::pair<int, int> e03 = (v_indices[0] < v_indices[3]) ? std::make_pair(v_indices[0], v_indices[3]) : std::make_pair(v_indices[3], v_indices[0]);
                if (ev_map.find(e03) == ev_map.end())
                {
                    std::vector<double> interpolated_point = interpolate(point_positions_.data() + v_indices[0] * 3, point_positions_.data() + v_indices[3] * 3, point_values_[v_indices[0]], point_values_[v_indices[3]]);
                    ev_map.insert({e03, mesh_vertices.size()});
                    mesh_vertices.push_back(interpolated_point);
                }
            }
            if (point_values_[v_indices[1]] * point_values_[v_indices[2]] < 0)
            {
                flag = true;
                std::pair<int, int> e12 = (v_indices[1] < v_indices[2]) ? std::make_pair(v_indices[1], v_indices[2]) : std::make_pair(v_indices[2], v_indices[1]);
                if (ev_map.find(e12) == ev_map.end())
                {
                    std::vector<double> interpolated_point = interpolate(point_positions_.data() + v_indices[1] * 3, point_positions_.data() + v_indices[2] * 3, point_values_[v_indices[1]], point_values_[v_indices[2]]);
                    ev_map.insert({e12, mesh_vertices.size()});
                    mesh_vertices.push_back(interpolated_point);
                }
            }
            if (point_values_[v_indices[1]] * point_values_[v_indices[3]] < 0)
            {
                flag = true;
                std::pair<int, int> e13 = (v_indices[1] < v_indices[3]) ? std::make_pair(v_indices[1], v_indices[3]) : std::make_pair(v_indices[3], v_indices[1]);
                if (ev_map.find(e13) == ev_map.end())
                {
                    std::vector<double> interpolated_point = interpolate(point_positions_.data() + v_indices[1] * 3, point_positions_.data() + v_indices[3] * 3, point_values_[v_indices[1]], point_values_[v_indices[3]]);
                    ev_map.insert({e13, mesh_vertices.size()});
                    mesh_vertices.push_back(interpolated_point);
                }
            }
            if (point_values_[v_indices[2]] * point_values_[v_indices[3]] < 0)
            {
                flag = true;
                std::pair<int, int> e23 = (v_indices[2] < v_indices[3]) ? std::make_pair(v_indices[2], v_indices[3]) : std::make_pair(v_indices[3], v_indices[2]);
                if (ev_map.find(e23) == ev_map.end())
                {
                    std::vector<double> interpolated_point = interpolate(point_positions_.data() + v_indices[2] * 3, point_positions_.data() + v_indices[3] * 3, point_values_[v_indices[2]], point_values_[v_indices[3]]);
                    ev_map.insert({e23, mesh_vertices.size()});
                    mesh_vertices.push_back(interpolated_point);
                }
            }

            if (flag)
            {
                intersection_cell.push_back(i);
            }
        }
        for (int i = 0; i < intersection_cell.size(); i++)
        {
            int cell_index = intersection_cell[i];

            unsigned char index = 0;
            std::vector<int> v_indices;
            for (index_t lv = 0; lv < 4; ++lv)
            {
                int v = delaunay_->cell_vertex(cell_index, lv);
                v_indices.push_back(int(v));

                if (point_values_[v] < 0)
                {
                    index |= (1 << lv);
                }
            }

            std::pair<int, int> e01 = (v_indices[0] < v_indices[1]) ? std::make_pair(v_indices[0], v_indices[1]) : std::make_pair(v_indices[1], v_indices[0]);
            std::pair<int, int> e02 = (v_indices[0] < v_indices[2]) ? std::make_pair(v_indices[0], v_indices[2]) : std::make_pair(v_indices[2], v_indices[0]);
            std::pair<int, int> e03 = (v_indices[0] < v_indices[3]) ? std::make_pair(v_indices[0], v_indices[3]) : std::make_pair(v_indices[3], v_indices[0]);
            std::pair<int, int> e12 = (v_indices[1] < v_indices[2]) ? std::make_pair(v_indices[1], v_indices[2]) : std::make_pair(v_indices[2], v_indices[1]);
            std::pair<int, int> e13 = (v_indices[1] < v_indices[3]) ? std::make_pair(v_indices[1], v_indices[3]) : std::make_pair(v_indices[3], v_indices[1]);
            std::pair<int, int> e23 = (v_indices[2] < v_indices[3]) ? std::make_pair(v_indices[2], v_indices[3]) : std::make_pair(v_indices[3], v_indices[2]);

            int v00, v10, v20, v01, v11, v21;

            // Case analysis
            switch (index)
            {
                // skip inside or outside situitions
            case 0x00:
            case 0x0F:
                break;
                // only vert 0 is inside
            case 0x01:
                v00 = ev_map[e01];
                v10 = ev_map[e03];
                v20 = ev_map[e02];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // only vert 1 is inside
            case 0x02:
                v00 = ev_map[e01];
                v10 = ev_map[e12];
                v20 = ev_map[e13];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // only vert 2 is inside
            case 0x04:
                v00 = ev_map[e02];
                v10 = ev_map[e23];
                v20 = ev_map[e12];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // only vert 3 is inside
            case 0x08:
                v00 = ev_map[e13];
                v10 = ev_map[e23];
                v20 = ev_map[e03];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // verts 0, 1 are inside
            case 0x03:
                v00 = ev_map[e03];
                v10 = ev_map[e02];
                v20 = ev_map[e13];

                mesh_faces.push_back({v00, v20, v10});
                v01 = ev_map[e02];
                v11 = ev_map[e12];
                v21 = ev_map[e13];

                mesh_faces.push_back({v01, v21, v11});
                break;

                // verts 0, 2 are inside
            case 0x05:
                v00 = ev_map[e03];
                v10 = ev_map[e12];
                v20 = ev_map[e01];

                mesh_faces.push_back({v00, v20, v10});
                v01 = ev_map[e12];
                v11 = ev_map[e03];
                v21 = ev_map[e23];

                mesh_faces.push_back({v01, v21, v11});
                break;

                // verts 0, 3 are inside
            case 0x09:
                v00 = ev_map[e01];
                v10 = ev_map[e13];
                v20 = ev_map[e02];

                mesh_faces.push_back({v00, v20, v10});
                v01 = ev_map[e13];
                v11 = ev_map[e23];
                v21 = ev_map[e02];

                mesh_faces.push_back({v01, v21, v11});
                break;

                // verts 1, 2 are inside
            case 0x06:
                v00 = ev_map[e01];
                v10 = ev_map[e02];
                v20 = ev_map[e13];

                mesh_faces.push_back({v00, v20, v10});
                v01 = ev_map[e13];
                v11 = ev_map[e02];
                v21 = ev_map[e23];
                mesh_faces.push_back({v01, v21, v11});
                break;

                // verts 2, 3 are inside
            case 0x0C:
                v00 = ev_map[e13];
                v10 = ev_map[e02];
                v20 = ev_map[e03];

                mesh_faces.push_back({v00, v20, v10});
                v01 = ev_map[e02];
                v11 = ev_map[e13];
                v21 = ev_map[e12];
                mesh_faces.push_back({v01, v21, v11});
                break;

                // verts 1, 3 are inside
            case 0x0A:
                v00 = ev_map[e03];
                v10 = ev_map[e01];
                v20 = ev_map[e12];

                mesh_faces.push_back({v00, v20, v10});
                v01 = ev_map[e12];
                v11 = ev_map[e23];
                v21 = ev_map[e03];
                mesh_faces.push_back({v01, v21, v11});
                break;

                // verts 0, 1, 2 are inside
            case 0x07:
                v00 = ev_map[e03];
                v10 = ev_map[e23];
                v20 = ev_map[e13];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // verts 0, 1, 3 are inside
            case 0x0B:
                v00 = ev_map[e12];
                v10 = ev_map[e23];
                v20 = ev_map[e02];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // verts 0, 2, 3 are inside
            case 0x0D:
                v00 = ev_map[e01];
                v10 = ev_map[e13];
                v20 = ev_map[e12];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // verts 1, 2, 3 are inside
            case 0x0E:
                v00 = ev_map[e01];
                v10 = ev_map[e02];
                v20 = ev_map[e03];

                mesh_faces.push_back({v00, v20, v10});
                break;

            default:
                // assert(false);
                continue;
            }
        }

        std::ofstream outfile(filename); // create a file named "example.txt"

        if (outfile.is_open())
        {
            for (size_t k = 0; k < mesh_vertices.size(); ++k)
            {
                std::vector<double> site = mesh_vertices[k];
                size_t n = site.size();
                outfile << "v ";
                for (size_t i = 0; i < site.size(); i++)
                {
                    outfile << site[i] << " ";
                }
                outfile << " \n";
            }

            for (size_t j = 0; j < mesh_faces.size(); j++)
            {
                std::vector<int> vertices_idx = mesh_faces[j];

                outfile << "f " << vertices_idx[0] + 1 << " " << vertices_idx[1] + 1 << " " << vertices_idx[2] + 1 << " \n";
            }
        }
    }

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> 
    MCMT::get_triangle_mesh() {

        std::vector<std::vector<double>> mesh_vertices;
        std::vector<std::vector<int>> mesh_faces;

        std::map<std::pair<int, int>, int> ev_map;
        int v_counter = 0;
        std::vector<int> intersection_cell;

        for (int i = 0; i < delaunay_->nb_finite_cells(); i++)
        {
            std::vector<int> v_indices;
            bool flag = false;
            for (index_t lv = 0; lv < 4; ++lv)
            {
                int v = delaunay_->cell_vertex(i, lv);
                v_indices.push_back(int(v));
            }

            if (point_values_[v_indices[0]] * point_values_[v_indices[1]] < 0)
            {
                flag = true;
                std::pair<int, int> e01 = (v_indices[0] < v_indices[1]) ? std::make_pair(v_indices[0], v_indices[1]) : std::make_pair(v_indices[1], v_indices[0]);
                if (ev_map.find(e01) == ev_map.end())
                {
                    std::vector<double> interpolated_point = interpolate(point_positions_.data() + v_indices[0] * 3, point_positions_.data() + v_indices[1] * 3, point_values_[v_indices[0]], point_values_[v_indices[1]]);
                    ev_map.insert({e01, mesh_vertices.size()});
                    mesh_vertices.push_back(interpolated_point);
                }
            }
            if (point_values_[v_indices[0]] * point_values_[v_indices[2]] < 0)
            {
                flag = true;
                std::pair<int, int> e02 = (v_indices[0] < v_indices[2]) ? std::make_pair(v_indices[0], v_indices[2]) : std::make_pair(v_indices[2], v_indices[0]);
                if (ev_map.find(e02) == ev_map.end())
                {
                    std::vector<double> interpolated_point = interpolate(point_positions_.data() + v_indices[0] * 3, point_positions_.data() + v_indices[2] * 3, point_values_[v_indices[0]], point_values_[v_indices[2]]);
                    ev_map.insert({e02, mesh_vertices.size()});
                    mesh_vertices.push_back(interpolated_point);
                }
            }
            if (point_values_[v_indices[0]] * point_values_[v_indices[3]] < 0)
            {
                flag = true;
                std::pair<int, int> e03 = (v_indices[0] < v_indices[3]) ? std::make_pair(v_indices[0], v_indices[3]) : std::make_pair(v_indices[3], v_indices[0]);
                if (ev_map.find(e03) == ev_map.end())
                {
                    std::vector<double> interpolated_point = interpolate(point_positions_.data() + v_indices[0] * 3, point_positions_.data() + v_indices[3] * 3, point_values_[v_indices[0]], point_values_[v_indices[3]]);
                    ev_map.insert({e03, mesh_vertices.size()});
                    mesh_vertices.push_back(interpolated_point);
                }
            }
            if (point_values_[v_indices[1]] * point_values_[v_indices[2]] < 0)
            {
                flag = true;
                std::pair<int, int> e12 = (v_indices[1] < v_indices[2]) ? std::make_pair(v_indices[1], v_indices[2]) : std::make_pair(v_indices[2], v_indices[1]);
                if (ev_map.find(e12) == ev_map.end())
                {
                    std::vector<double> interpolated_point = interpolate(point_positions_.data() + v_indices[1] * 3, point_positions_.data() + v_indices[2] * 3, point_values_[v_indices[1]], point_values_[v_indices[2]]);
                    ev_map.insert({e12, mesh_vertices.size()});
                    mesh_vertices.push_back(interpolated_point);
                }
            }
            if (point_values_[v_indices[1]] * point_values_[v_indices[3]] < 0)
            {
                flag = true;
                std::pair<int, int> e13 = (v_indices[1] < v_indices[3]) ? std::make_pair(v_indices[1], v_indices[3]) : std::make_pair(v_indices[3], v_indices[1]);
                if (ev_map.find(e13) == ev_map.end())
                {
                    std::vector<double> interpolated_point = interpolate(point_positions_.data() + v_indices[1] * 3, point_positions_.data() + v_indices[3] * 3, point_values_[v_indices[1]], point_values_[v_indices[3]]);
                    ev_map.insert({e13, mesh_vertices.size()});
                    mesh_vertices.push_back(interpolated_point);
                }
            }
            if (point_values_[v_indices[2]] * point_values_[v_indices[3]] < 0)
            {
                flag = true;
                std::pair<int, int> e23 = (v_indices[2] < v_indices[3]) ? std::make_pair(v_indices[2], v_indices[3]) : std::make_pair(v_indices[3], v_indices[2]);
                if (ev_map.find(e23) == ev_map.end())
                {
                    std::vector<double> interpolated_point = interpolate(point_positions_.data() + v_indices[2] * 3, point_positions_.data() + v_indices[3] * 3, point_values_[v_indices[2]], point_values_[v_indices[3]]);
                    ev_map.insert({e23, mesh_vertices.size()});
                    mesh_vertices.push_back(interpolated_point);
                }
            }

            if (flag)
            {
                intersection_cell.push_back(i);
            }
        }
        for (int i = 0; i < intersection_cell.size(); i++)
        {
            int cell_index = intersection_cell[i];

            unsigned char index = 0;
            std::vector<int> v_indices;
            for (index_t lv = 0; lv < 4; ++lv)
            {
                int v = delaunay_->cell_vertex(cell_index, lv);
                v_indices.push_back(int(v));

                if (point_values_[v] < 0)
                {
                    index |= (1 << lv);
                }
            }

            std::pair<int, int> e01 = (v_indices[0] < v_indices[1]) ? std::make_pair(v_indices[0], v_indices[1]) : std::make_pair(v_indices[1], v_indices[0]);
            std::pair<int, int> e02 = (v_indices[0] < v_indices[2]) ? std::make_pair(v_indices[0], v_indices[2]) : std::make_pair(v_indices[2], v_indices[0]);
            std::pair<int, int> e03 = (v_indices[0] < v_indices[3]) ? std::make_pair(v_indices[0], v_indices[3]) : std::make_pair(v_indices[3], v_indices[0]);
            std::pair<int, int> e12 = (v_indices[1] < v_indices[2]) ? std::make_pair(v_indices[1], v_indices[2]) : std::make_pair(v_indices[2], v_indices[1]);
            std::pair<int, int> e13 = (v_indices[1] < v_indices[3]) ? std::make_pair(v_indices[1], v_indices[3]) : std::make_pair(v_indices[3], v_indices[1]);
            std::pair<int, int> e23 = (v_indices[2] < v_indices[3]) ? std::make_pair(v_indices[2], v_indices[3]) : std::make_pair(v_indices[3], v_indices[2]);

            int v00, v10, v20, v01, v11, v21;

            // Case analysis
            switch (index)
            {
                // skip inside or outside situitions
            case 0x00:
            case 0x0F:
                break;
                // only vert 0 is inside
            case 0x01:
                v00 = ev_map[e01];
                v10 = ev_map[e03];
                v20 = ev_map[e02];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // only vert 1 is inside
            case 0x02:
                v00 = ev_map[e01];
                v10 = ev_map[e12];
                v20 = ev_map[e13];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // only vert 2 is inside
            case 0x04:
                v00 = ev_map[e02];
                v10 = ev_map[e23];
                v20 = ev_map[e12];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // only vert 3 is inside
            case 0x08:
                v00 = ev_map[e13];
                v10 = ev_map[e23];
                v20 = ev_map[e03];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // verts 0, 1 are inside
            case 0x03:
                v00 = ev_map[e03];
                v10 = ev_map[e02];
                v20 = ev_map[e13];

                mesh_faces.push_back({v00, v20, v10});
                v01 = ev_map[e02];
                v11 = ev_map[e12];
                v21 = ev_map[e13];

                mesh_faces.push_back({v01, v21, v11});
                break;

                // verts 0, 2 are inside
            case 0x05:
                v00 = ev_map[e03];
                v10 = ev_map[e12];
                v20 = ev_map[e01];

                mesh_faces.push_back({v00, v20, v10});
                v01 = ev_map[e12];
                v11 = ev_map[e03];
                v21 = ev_map[e23];

                mesh_faces.push_back({v01, v21, v11});
                break;

                // verts 0, 3 are inside
            case 0x09:
                v00 = ev_map[e01];
                v10 = ev_map[e13];
                v20 = ev_map[e02];

                mesh_faces.push_back({v00, v20, v10});
                v01 = ev_map[e13];
                v11 = ev_map[e23];
                v21 = ev_map[e02];

                mesh_faces.push_back({v01, v21, v11});
                break;

                // verts 1, 2 are inside
            case 0x06:
                v00 = ev_map[e01];
                v10 = ev_map[e02];
                v20 = ev_map[e13];

                mesh_faces.push_back({v00, v20, v10});
                v01 = ev_map[e13];
                v11 = ev_map[e02];
                v21 = ev_map[e23];
                mesh_faces.push_back({v01, v21, v11});
                break;

                // verts 2, 3 are inside
            case 0x0C:
                v00 = ev_map[e13];
                v10 = ev_map[e02];
                v20 = ev_map[e03];

                mesh_faces.push_back({v00, v20, v10});
                v01 = ev_map[e02];
                v11 = ev_map[e13];
                v21 = ev_map[e12];
                mesh_faces.push_back({v01, v21, v11});
                break;

                // verts 1, 3 are inside
            case 0x0A:
                v00 = ev_map[e03];
                v10 = ev_map[e01];
                v20 = ev_map[e12];

                mesh_faces.push_back({v00, v20, v10});
                v01 = ev_map[e12];
                v11 = ev_map[e23];
                v21 = ev_map[e03];
                mesh_faces.push_back({v01, v21, v11});
                break;

                // verts 0, 1, 2 are inside
            case 0x07:
                v00 = ev_map[e03];
                v10 = ev_map[e23];
                v20 = ev_map[e13];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // verts 0, 1, 3 are inside
            case 0x0B:
                v00 = ev_map[e12];
                v10 = ev_map[e23];
                v20 = ev_map[e02];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // verts 0, 2, 3 are inside
            case 0x0D:
                v00 = ev_map[e01];
                v10 = ev_map[e13];
                v20 = ev_map[e12];

                mesh_faces.push_back({v00, v20, v10});
                break;

                // verts 1, 2, 3 are inside
            case 0x0E:
                v00 = ev_map[e01];
                v10 = ev_map[e02];
                v20 = ev_map[e03];

                mesh_faces.push_back({v00, v20, v10});
                break;

            default:
                // assert(false);
                continue;
            }
        }
        return std::make_pair(mesh_vertices, mesh_faces);
    }

    void MCMT::save_grid_mesh(std::string filename, float x_clip_plane)
    {

        std::vector<std::vector<double>> mesh_vertices;
        std::vector<std::vector<int>> mesh_faces;

        for (int i = 0; i < delaunay_->nb_vertices(); i++)
        {
            double x = delaunay_->vertex_ptr(i)[0];
            double y = delaunay_->vertex_ptr(i)[1];
            double z = delaunay_->vertex_ptr(i)[2];
            mesh_vertices.push_back(std::vector<double>{x, y, z});
        }

        for (int i = 0; i < delaunay_->nb_finite_cells(); i++)
        {
            std::vector<int> v_indices;
            bool flag = false;
            for (index_t lv = 0; lv < 4; ++lv)
            {
                int v = delaunay_->cell_vertex(i, lv);
                double x = delaunay_->vertex_ptr(v)[0];

                if (x < x_clip_plane)
                {
                    flag = true;
                    break;
                }
                v_indices.push_back(int(v));
            }
            if (flag)
                continue;
            mesh_faces.push_back(v_indices);
        }

        std::ofstream outfile(filename); // create a file named "example.txt"

        if (outfile.is_open())
        {
            for (size_t k = 0; k < mesh_vertices.size(); ++k)
            {
                std::vector<double> site = mesh_vertices[k];
                size_t n = site.size();
                outfile << "v ";
                for (size_t i = 0; i < site.size(); i++)
                {
                    outfile << site[i] << " ";
                }
                outfile << " \n";
            }

            for (size_t j = 0; j < mesh_faces.size(); j++)
            {
                std::vector<int> vertices_idx = mesh_faces[j];

                if (vertices_idx.size() == 4)
                {
                    outfile << "f " << vertices_idx[0] + 1 << " " << vertices_idx[2] + 1 << " " << vertices_idx[1] + 1 << " \n";
                    outfile << "f " << vertices_idx[0] + 1 << " " << vertices_idx[3] + 1 << " " << vertices_idx[2] + 1 << " \n";
                    outfile << "f " << vertices_idx[0] + 1 << " " << vertices_idx[1] + 1 << " " << vertices_idx[3] + 1 << " \n";
                    outfile << "f " << vertices_idx[1] + 1 << " " << vertices_idx[2] + 1 << " " << vertices_idx[3] + 1 << " \n";
                }
            }
        }
    }

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> 
    MCMT::get_grid_mesh(float x_clip_plane)
    {

        std::vector<std::vector<double>> mesh_vertices;
        std::vector<std::vector<int>> mesh_faces;

        for (int i = 0; i < delaunay_->nb_vertices(); i++)
        {
            double x = delaunay_->vertex_ptr(i)[0];
            double y = delaunay_->vertex_ptr(i)[1];
            double z = delaunay_->vertex_ptr(i)[2];
            mesh_vertices.push_back(std::vector<double>{x, y, z});
        }

        for (int i = 0; i < delaunay_->nb_finite_cells(); i++)
        {
            std::vector<int> v_indices;
            bool flag = false;
            for (index_t lv = 0; lv < 4; ++lv)
            {
                int v = delaunay_->cell_vertex(i, lv);
                double x = delaunay_->vertex_ptr(v)[0];

                if (x < x_clip_plane)
                {
                    flag = true;
                    break;
                }
                v_indices.push_back(int(v));
            }
            if (flag)
                continue;
            mesh_faces.push_back(v_indices);
        }
        return std::make_pair(mesh_vertices, mesh_faces);

    }

}
