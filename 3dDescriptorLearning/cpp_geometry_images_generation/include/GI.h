/************************************************************
* Author: Hanyu Wang(王涵玉)
************************************************************/

#ifndef GI_H
#define GI_H

#include <iostream>
#include <vector>
#include "GPC.h"
#include "Vector3.h"
#include "utils.h"

namespace GIGen
{
    //template <typename Mesh>
    typedef MeshOM Mesh;
    class GI
    {
    private:
        const GPC<Mesh> &gpc;
        const int gi_size, rotation_num, used_attr_num;
        std::vector<std::vector<std::vector<std::vector<float>>>> geo_img_all_rots; // Geometry images of all rotation angles.


        inline double triangle_area(const Point_2& p1, const Point_2& p2, const Point_2& p3)
        {
            double ax = p2.x() - p1.x();
            double ay = p2.y() - p1.y();
            double bx = p3.x() - p1.x();
            double by = p3.y() - p1.y();
            return fabs(0.5 * (ax * by - ay * bx));
        }


        bool append_features(const std::vector<std::vector<Point_2>>& sampling_points, const unsigned int &gi_idx, bool with_hks=false)
        {
            auto &gi = geo_img_all_rots[gi_idx];

            for (int r = 0; r < gi_size; r++)
            {
                for (int c = 0; c < gi_size; c++)
                {
                    auto &this_point = sampling_points[r][c];
                    auto idx_vec = gpc.find_triangle_vertices(this_point);

                    auto &&p0 = gpc.access_vertex_by_mesh_idx(idx_vec[0]);
                    auto &&p1 = gpc.access_vertex_by_mesh_idx(idx_vec[1]);
                    auto &&p2 = gpc.access_vertex_by_mesh_idx(idx_vec[2]);

                    double s0 = triangle_area(this_point, p1, p2);
                    double s1 = triangle_area(p0, this_point, p2);
                    double s2 = triangle_area(p0, p1, this_point);
                    double s = s0 + s1 + s2;

                    double p0_c_max, p1_c_max, p2_c_max;
                    double p0_c_min, p1_c_min, p2_c_min;
                    Point p0_normal, p1_normal, p2_normal;


                    p0.get_attr("c_max", p0_c_max);
                    p0.get_attr("c_min", p0_c_min);
                    p0.get_attr("normal", p0_normal);

                    p1.get_attr("c_max", p1_c_max);
                    p1.get_attr("c_min", p1_c_min);
                    p1.get_attr("normal", p1_normal);

                    p2.get_attr("c_max", p2_c_max);
                    p2.get_attr("c_min", p2_c_min);
                    p2.get_attr("normal", p2_normal);


                    gi[r][c].emplace_back((p0_c_max * s0 + p1_c_max * s1 + p2_c_max * s2) / s);
                    gi[r][c].emplace_back((p0_c_min * s0 + p1_c_min * s1 + p2_c_min * s2) / s);
                    gi[r][c].emplace_back((p0_normal.x() * s0 + p1_normal.x() * s1 + p2_normal.x() * s2) / s);
                    gi[r][c].emplace_back((p0_normal.y() * s0 + p1_normal.y() * s1 + p2_normal.y() * s2) / s);
                    gi[r][c].emplace_back((p0_normal.z() * s0 + p1_normal.z() * s1 + p2_normal.z() * s2) / s);

                    if (with_hks)
                    {
                        std::vector<double> p0_hks, p1_hks, p2_hks;
                        p0.get_attr("hks", p0_hks);
                        p1.get_attr("hks", p1_hks);
                        p2.get_attr("hks", p2_hks);

                        for (int i = 0; i < p0_hks.size(); i++)
                        {
                            gi[r][c].emplace_back((p0_hks[i] * s0 + p1_hks[i] * s1 + p2_hks[i] * s2) / s);
                        }
                    }

                }
            }

            return true;

        }



    public:

        GI(const GPC<Mesh> &gpc, const std::vector<double> &max_radius, const int &gi_size, const int &rotation_num, const int &used_attr_num = 5) :
            gpc(gpc),
            geo_img_all_rots(rotation_num, std::vector<std::vector<std::vector<float>>>(gi_size, std::vector<std::vector<float>>(gi_size))),
            gi_size(gi_size), rotation_num(rotation_num), used_attr_num(used_attr_num)
        {
            if (!gpc.point_num())
                return;

            // Initialization of the geometry image;
            for (auto &gi : this->geo_img_all_rots)
            {
                for (auto &c : gi)
                {
                    for (auto &p : c)
                    {
                        p.reserve(used_attr_num * max_radius.size());
                    }
                }
            }


            double start_x = -sqrt(2) / 2 + sqrt(2) / (2 * double(gi_size));
            double start_y = sqrt(2) / 2 - sqrt(2) / (2 * double(gi_size));
            double delta = sqrt(2) / (double(gi_size));


            //Sampling points
            std::vector<std::vector<Point_2>> generic_sampling_points(gi_size, std::vector<Point_2>(gi_size));
            for (int r = 0; r < gi_size; r++)
            {
                for (int c = 0; c < gi_size; c++)
                {
                    generic_sampling_points[r][c] = Point_2(start_x + c * delta, start_y - r * delta);
                }
            }



            double rotation_rad = 2 * M_PI / rotation_num;
            for (unsigned int i = 0; i < rotation_num; i++)
            {
                double rad = rotation_rad * i;

                for (double radius : max_radius)
                {
                    std::vector<std::vector<Point_2>> sampling_points = generic_sampling_points;

                    for (auto& row : sampling_points)
                    {
                        for (auto& point : row)
                        {
                            double x = point.x() * cos(rad) - point.y() * sin(rad); // Rotate the sampling points.
                            double y = point.x() * sin(rad) + point.y() * cos(rad);
                            point = Point_2(x * radius, y * radius); // Scale the sampling points to fit parameterization radius.
                        }
                    }

                    
                    if (radius == max_radius[max_radius.size() - 1])
                    {
                        this->append_features(sampling_points, i, true);
                    }
                    else
                    {
                        this->append_features(sampling_points, i, false);
                    }
                }


            }
        }


        bool save_all(const std::string &geo_img_dir, const std::string& name_prefix) const
        {
            return this->save_all(Dir(geo_img_dir), name_prefix);

        }

        bool save_all(const Dir &geo_img_dir, const std::string& name_prefix) const
        {
            for (unsigned int i = 0; i < this->rotation_num; i++)
            {
                auto geo_img_path = geo_img_dir + name_prefix + "_rot_" + to_string_f("%02d", i) + ".gi";

                std::ofstream out(geo_img_path);

                int count = 100;
                while (!out && count > 0)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    out = std::ofstream(geo_img_path);
                    count--;
                }
                if (!out) return false;

                for (int ch = 0; ch < geo_img_all_rots[i][0][0].size(); ch++)
                {
                    for (auto &r_vec : geo_img_all_rots[i])
                    {
                        for (auto &val : r_vec)
                            out << std::fixed << val[ch] << " ";
                        out << std::endl;
                    }
                    out << std::endl;
                }

                out.close();
            }
            return true;
        }

        bool save_all_rotation_in_one(const std::string &geo_img_dir, const std::string& name_prefix) const
        {
            auto geo_img_path = geo_img_dir + name_prefix + ".gi";

            std::ofstream out(geo_img_path);
            int count = 100;
            while (!out && count > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                out = std::ofstream(geo_img_path);
                count--;
            }
            if (!out) return false;

            for (unsigned int i = 0; i < this->rotation_num; i++)
            {
                for (int ch = 0; ch < geo_img_all_rots[i][0][0].size(); ch++)
                {
                    for (auto &r_vec : geo_img_all_rots[i])
                    {
                        for (auto &val : r_vec)
                            out << std::fixed << val[ch] << " ";
                        out << std::endl;
                    }
                    out << std::endl;
                }

                out << std::endl << std::endl;

            }

            out.close();
            return true;
        }


    };


}; // End namespace GIGen


#endif // !GI_H
