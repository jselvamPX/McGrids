
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <math.h>
#include <random>
#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/stopwatch.h>
#include <geogram/basic/file_system.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_reorder.h>
#include <geogram/delaunay/delaunay.h>
#include <geogram/delaunay/periodic_delaunay_3d.h>
#include <geogram/voronoi/RVD.h>
#include <geogram/voronoi/generic_RVD.h>
#include <geogram/voronoi/integration_simplex.h>
#include <geogram/voronoi/convex_cell.h>
#include <algorithm>

namespace GEO
{

	class MCMT
	{
	public:
		MCMT();
		~MCMT();

		void clear();

		void add_points(int num_points, double *point_positions, double *point_values);
		void add_mid_points(int num_points, double *point_positions, double *point_values);
		std::vector<double> get_mid_points();
		std::vector<double> get_grid_points();
		std::vector<int> get_grids();
		std::vector<double> sample_points_rejection(int num_samples, double min_value, double max_value);
		// std::vector<double> sample_points(int num_samples);
		std::vector<double> lloyd_relaxation(double *point_positions, int num_points, int num_iter);
		void output_grid_points(std::string filename);
		void save_triangle_mesh(std::string filename);
		void save_grid_mesh(std::string filename, float x_clip_plane);
		std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> get_triangle_mesh();
		std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> get_grid_mesh(float x_clip_plane);

		std::vector<double> sample_points_voronoi(const int num_points);

	private:
		PeriodicDelaunay3d *delaunay_;
		// PeriodicDelaunay3d::IncidentTetrahedra W_;
		bool periodic_ = false;
		double max_bound = 0;
		double min_bound = 0;
		int num_point_visited_ = 0;
		std::vector<double> point_positions_;
		std::vector<double> point_values_;
		std::vector<double> point_errors_;
		std::vector<double> point_volumes_;
		std::vector<bool> volume_changed_;

		std::vector<double> sample_tet(std::vector<double> point_positions);
		std::vector<double> compute_tet_error();
		std::vector<double> sample_polytope(int vertex_index);
		std::vector<double> compute_voronoi_error();

		double tetrahedronVolume(const std::vector<double> &coordinates);
		void save_face(std::ofstream &output_mesh, const std::vector<double> &points, int &vertex_count);
		index_t nb_points() const
		{
			return index_t(point_positions_.size() / 3);
		}
		void get_cell(index_t v, ConvexCell &C, PeriodicDelaunay3d::IncidentTetrahedra& W);

		std::vector<double> compute_face_mid_point(int num_points, const std::vector<double> &points);
		std::vector<double> interpolate(double *point1, double *point2, double sd1, double sd2);
	};
}