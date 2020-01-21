function [obj_mask, depth_map] = segmentation_mask(proj_pts, classes, faces, h, w, obj_mask, depth_map)
%SEGMENTATION_MASK 
%   Creates a segmentation mask

%% Add class label to each point
proj_pts(:,4) = classes;

%% Build projected triangles
num_faces = length(faces);
proj_faces = zeros(num_faces, 6);
depth_vals = zeros(num_faces, 3);
label_vals = zeros(num_faces, 3);
for i = 1:num_faces
    f = faces(i,:);
    
    % Face vertices
    v0 = proj_pts(f(1), :);
    v1 = proj_pts(f(2), :);
    v2 = proj_pts(f(3), :);
    
    % Projected face coordinates
    proj_faces(i, :) = [v0(1:2) v1(1:2) v2(1:2)];
    
    % Depth values
    depth_vals(i, :) = [v0(3) v1(3) v2(3)];
    
    % Label values
    label_vals(i, :) = [v0(4) v1(4) v2(4)];
end

%% Render all triangles
for i = 1:num_faces
    v0 = round(proj_faces(i, 1:2));
    v1 = round(proj_faces(i, 3:4));
    v2 = round(proj_faces(i, 5:6));
    z = depth_vals(i, :);
    l = label_vals(i, :);
    
    % Create bounding box
    maxX = max(v0(1), max(v1(1), v2(1))); maxX = clamp(1, maxX, w);
    minX = min(v0(1), min(v1(1), v2(1))); minX = clamp(1, minX, w);
    maxY = max(v0(2), max(v1(2), v2(2))); maxY = clamp(1, maxY, h);
    minY = min(v0(2), min(v1(2), v2(2))); minY = clamp(1, minY, h);
    
    area = edge_f(v0, v1, v2);
    
    % Check which pixels are inside the triangle using edge function
    for x = minX:maxX
        for y = minY:maxY
            p = [x y];
            
            w0 = edge_f(v1, v2, p);
            w1 = edge_f(v2, v0, p);
            w2 = edge_f(v0, v1, p);
            
            d0 = pt_dist(v0, p);
            d1 = pt_dist(v1, p);
            d2 = pt_dist(v2, p);
            min_d = min(d0, min(d1, d2));
            
            if (w0 >= 0 && w1 >= 0 && w2 >= 0 && area > 0)
                % Convert to barycentric coordinates to compute z value
                w0 = w0 / area;
                w1 = w1 / area;
                w2 = w2 / area;
                
                % Z-buffer
                z_old = depth_map(y, x);
                z_new = z(1) * w0 + z(2) * w1 + z(3) * w2;
                
                % Label
                if min_d == d0
                    l_new = l(1);
                elseif min_d == d1
                    l_new = l(2);
                else
                    l_new = l(3);
                end
                
                % Set pixel class and depth
                if z_old == 0
                    depth_map(y,x) = z_new;
                    obj_mask(y,x) = l_new;
                elseif z_new < z_old 
                    depth_map(y, x) = z_new;
                    obj_mask(y, x) = l_new;
                end
            end
        end
    end    
end

end

