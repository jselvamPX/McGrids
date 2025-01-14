#include <torch/extension.h>

#include "fast_mcmt.hpp"


namespace mcmt
{
  GEO::MCMT mcmt = GEO::MCMT();

    void add_points(torch::Tensor point_positions, torch::Tensor point_values)
    {
        // Ensure the tensors are on CPU and are of type double
        if (!point_positions.device().is_cpu() || !point_values.device().is_cpu())
        {
            throw std::runtime_error("Input tensors must be on CPU");
        }
        if (point_positions.scalar_type() != torch::kDouble || point_values.scalar_type() != torch::kDouble)
        {
            throw std::runtime_error("Input tensors must be of type torch.double (double)");
        }

        int64_t num_points = point_positions.size(0);
        int64_t num_values = point_values.size(0);

        if (num_points != num_values)
        {
            throw std::runtime_error("Input points and values must have the same number of rows");
        }

        // Obtain pointers to the data
        double* point_positions_ptr = point_positions.data_ptr<double>();
        double* point_values_ptr = point_values.data_ptr<double>();

        mcmt.add_points(num_points, point_positions_ptr, point_values_ptr);
    }


    void add_mid_points(torch::Tensor point_positions, torch::Tensor point_values)
    {
        // Ensure the tensors are on CPU and are of type double
        if (!point_positions.device().is_cpu() || !point_values.device().is_cpu())
        {
            throw std::runtime_error("Input tensors must be on CPU");
        }
        if (point_positions.scalar_type() != torch::kDouble || point_values.scalar_type() != torch::kDouble)
        {
            throw std::runtime_error("Input tensors must be of type torch.double (double)");
        }

        int64_t num_points = point_positions.size(0);
        int64_t num_values = point_values.size(0);

        if (num_points != num_values)
        {
            throw std::runtime_error("Input points and values must have the same number of rows");
        }

        // Obtain pointers to the data
        double* point_positions_ptr = point_positions.data_ptr<double>();
        double* point_values_ptr = point_values.data_ptr<double>();

        mcmt.add_mid_points(num_points, point_positions_ptr, point_values_ptr);
    }

    torch::Tensor sample_points_rejection(int num_points, double min_value, double max_value)
    {
        std::vector<double> new_samples = mcmt.sample_points_rejection(num_points, min_value, max_value);

        // Create a tensor from the vector data
        torch::Tensor result = torch::from_blob(new_samples.data(), {(int64_t)new_samples.size()}, torch::kDouble);

        // Clone the tensor to own the data
        return result.clone();
    }

      torch::Tensor sample_points_voronoi(int num_points)
    {
        std::vector<double> new_samples = mcmt.sample_points_voronoi(num_points);
        torch::Tensor result = torch::from_blob(new_samples.data(), {(int64_t)new_samples.size()}, torch::kDouble);
        return result.clone();
    }

    torch::Tensor get_grid_points()
    {
        std::vector<double> grid_points = mcmt.get_grid_points();
        torch::Tensor result = torch::from_blob(grid_points.data(), {(int64_t)grid_points.size()}, torch::kDouble);
        return result.clone();
    }

    torch::Tensor lloyd_relaxation(torch::Tensor point_positions, int num_iter, double min_value, double max_value)
    {
        if (!point_positions.device().is_cpu())
        {
            throw std::runtime_error("Input tensor must be on CPU");
        }
        if (point_positions.scalar_type() != torch::kDouble)
        {
            throw std::runtime_error("Input tensor must be of type torch.double (double)");
        }

        int64_t num_points = point_positions.size(0);

        double* point_positions_ptr = point_positions.data_ptr<double>();

        std::vector<double> new_samples = mcmt.lloyd_relaxation(point_positions_ptr, num_points, num_iter);

        torch::Tensor result = torch::from_blob(new_samples.data(), {(int64_t)new_samples.size()}, torch::kDouble);
        return result.clone();
    }

    torch::Tensor get_mid_points()
    {
        std::vector<double> mid_points = mcmt.get_mid_points();
        torch::Tensor result = torch::from_blob(mid_points.data(), {(int64_t)mid_points.size()}, torch::kDouble);
        return result.clone();
    }

    torch::Tensor get_grids()
    {
        std::vector<int> grid = mcmt.get_grids();
        torch::Tensor result = torch::from_blob(grid.data(), {(int64_t)grid.size()}, torch::kInt);
        return result.clone();
    }

    void output_triangle_mesh(const std::string& filename)
    {
        mcmt.save_triangle_mesh(filename);
    }

    void output_grid_mesh(const std::string& filename, float x_clip_plane)
    {
        mcmt.save_grid_mesh(filename, x_clip_plane);
    }


    void clear_mcmt()
    {
        mcmt.clear();
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
    {
        m.def("add_points", &add_points, "Add points to MCMT");
        m.def("add_mid_points", &add_mid_points, "Add mid-points to MCMT");
        m.def("sample_points_rejection", &sample_points_rejection, "Sample points using rejection method");
        m.def("sample_points_voronoi", &sample_points_voronoi, "Sample points using Voronoi method");
        m.def("lloyd_relaxation", &lloyd_relaxation, "Perform Lloyd relaxation");
        m.def("get_grid_points", &get_grid_points, "Get grid points");
        m.def("get_mid_points", &get_mid_points, "Get mid-points");
        m.def("get_grids", &get_grids, "Get grids");
        m.def("output_triangle_mesh", &output_triangle_mesh, "Output triangle mesh");
        m.def("output_grid_mesh", &output_grid_mesh, "Output grid mesh");
        m.def("clear_mcmt", &clear_mcmt, "Clear MCMT");
        m.def("get_triangle_mesh", []() {
            auto [vertices, faces] = mcmt.get_triangle_mesh();
            
            // Convert vertices to torch tensor
            std::vector<float> vertices_flat;
            vertices_flat.reserve(vertices.size() * 3);
            for (const auto& v : vertices) {
                vertices_flat.push_back(v[0]);
                vertices_flat.push_back(v[1]);
                vertices_flat.push_back(v[2]);
            }
            
            // Convert faces to torch tensor
            std::vector<int64_t> faces_flat;
            faces_flat.reserve(faces.size() * 3);
            for (const auto& f : faces) {
                faces_flat.push_back(f[0]);
                faces_flat.push_back(f[1]);
                faces_flat.push_back(f[2]);
            }
            
            auto vertices_tensor = torch::from_blob(vertices_flat.data(), 
                                                {(int64_t)vertices.size(), 3}, 
                                                torch::kFloat).clone();
            
            auto faces_tensor = torch::from_blob(faces_flat.data(), 
                                            {(int64_t)faces.size(), 3}, 
                                            torch::kLong).clone();
            
            return std::make_tuple(vertices_tensor, faces_tensor);
        }, "Get triangle mesh as vertices and faces tensors");
        m.def("get_grid_mesh", [](float x_clip_plane) {
        auto [vertices, faces] = mcmt.get_grid_mesh(x_clip_plane);
        
        // Convert vertices to torch tensor
        std::vector<float> vertices_flat;
        vertices_flat.reserve(vertices.size() * 3);
        for (const auto& v : vertices) {
            vertices_flat.push_back(v[0]);
            vertices_flat.push_back(v[1]);
            vertices_flat.push_back(v[2]);
        }
        
        // Convert faces to torch tensor
        std::vector<int64_t> faces_flat;
        faces_flat.reserve(faces.size() * 3);
        for (const auto& f : faces) {
            faces_flat.push_back(f[0]);
            faces_flat.push_back(f[1]);
            faces_flat.push_back(f[2]);
        }
        
        auto vertices_tensor = torch::from_blob(vertices_flat.data(), 
                                            {(int64_t)vertices.size(), 3}, 
                                            torch::kFloat).clone();
        
        auto faces_tensor = torch::from_blob(faces_flat.data(), 
                                        {(int64_t)faces.size(), 3}, 
                                        torch::kLong).clone();
        
        return std::make_tuple(vertices_tensor, faces_tensor);
    }, "Get grid mesh as vertices and faces tensors");
        }
}
