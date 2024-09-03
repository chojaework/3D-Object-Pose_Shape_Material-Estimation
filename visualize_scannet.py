# script for scan2cad data collection
# save cropped raw point cloud

import os
import json
from tqdm import tqdm
import numpy as np
import open3d as o3d
import transforms3d
from utils.preprocess import path_dict, load_raw_pc
from utils.read_json import *
if __name__ == "__main__":
    # path
    scan_root = "/Users/tyzhao/Desktop/scans/"
    cad_root = "/Users/tyzhao/Desktop/workspace/data/ShapeNetCore.v2.PC15k"
    scan2cad = load_json("/Users/tyzhao/Desktop/scan2cad_download_link/full_annotations.json")
    target_dir = "/Users/tyzhao/Desktop"

    id2path = path_dict(cad_root)

    for idx in tqdm(range(len(scan2cad))):
        scannetid = scan2cad[idx]["id_scan"]

        if not scannetid == "scene0000_00":
            continue

        aggr = load_json(os.path.join(scan_root, scannetid, "{}_vh_clean.aggregation.json".format(scannetid)))
        
        scan_origin = o3d.io.read_point_cloud(os.path.join(scan_root, scannetid, "{}_vh_clean.ply".format(scannetid)), format='ply')

        scan_mesh = o3d.io.read_triangle_mesh(os.path.join(scan_root, scannetid, "{}_vh_clean.ply".format(scannetid)))

        scan_pose = scan2cad[idx]["trs"]

        scan_points = apply_trans(np.asarray(scan_origin.points), scan_pose["translation"], scan_pose["rotation"], scan_pose["scale"], mode="normal") 

        scan_rot = build_pcd(scan_points, np.asarray(scan_origin.colors))
        #o3d.visualization.draw_geometries([scan_rot])
        seg = load_json(os.path.join(scan_root, scannetid, "{}_vh_clean.segs.json".format(scannetid)))["segIndices"]
        seg = np.array(seg)


        centers = [];coords = [];colors = [];seg_indices = []
        for cat in aggr["segGroups"]:
            #print(cat["label"])
            coord = [];color = [];seg_indice = []
            for objid in cat["segments"]:
                #print(objid)
                seg_idx = np.nonzero(seg==objid)
                coord.append(np.asarray(scan_points)[seg_idx])
                seg_indice.append(seg_idx[0])
            coords.append(np.concatenate(coord, 0))
            centers.append(coords[-1].mean(0))
            seg_indices.append(np.concatenate(seg_indice, 0))

        centers = np.array(centers)

        for i, aligned_model in enumerate(scan2cad[idx]["aligned_models"]):

            cad_path = id2path[aligned_model["id_cad"]]
            print(i, aligned_model["catid_cad"], aligned_model["id_cad"])
            cad_pc = load_raw_pc(os.path.join(cad_root, cad_path), 5000)

            cad_pose = aligned_model["trs"]
            cad_pc = apply_trans(cad_pc, cad_pose["translation"], cad_pose["rotation"], cad_pose["scale"], mode="normal")

            dist = np.linalg.norm(centers-np.array(cad_pose["translation"]), 2, 1)
            nearest = np.argmin(dist)

            print(dist[nearest])
            
            seg_pcd = build_pcd(coords[nearest], np.array([0,1,0]))
            cad_pcd = build_pcd(cad_pc, np.array([1,0,0]))

            o3d.visualization.draw_geometries([cad_pcd, seg_pcd])

            mesh = o3d.geometry.TriangleMesh()
            seg_indice = seg_indices[nearest]

            mesh.vertices = o3d.utility.Vector3dVector(np.asarray(scan_mesh.vertices)[seg_indice])
            tri = np.asarray(scan_mesh.triangles)
            valid_vertex = np.isin (tri, seg_indice)
            valid_tri = np.logical_and(valid_vertex[:, 0], np.logical_and(valid_vertex[:, 1], valid_vertex[:, 2]))
            new_tri = convert_tri(tri[valid_tri], seg_indice)

            mesh.triangles = o3d.utility.Vector3iVector(new_tri)
            #o3d.visualization.draw_geometries([mesh])
            pcd = mesh.sample_points_uniformly(number_of_points=5000)
            #o3d.visualization.draw_geometries([cad_pcd, pcd])
            #np.save(os.path.join(target_dir, "{}.npy".format(i)), np.asarray(pcd.points))
