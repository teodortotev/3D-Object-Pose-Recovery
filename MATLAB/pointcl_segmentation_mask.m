function obj_mask = pointcl_segmentation_mask(proj_pts, faces, classes, h, w)
%SEGMENTATION_MASK
%   Creates a multi-class segmentation mask

% Round to nearest integer pixel
proj_pts(:,1:2) = round(proj_pts(:,1:2));

% Add the class label for each point
proj_pts(:,4) = classes;

% Sort the points according to x, y, and distance
proj_pts = sortrows(proj_pts, [1 2 3]);

% Initialize Segmentation Mask
obj_mask = zeros(h, w) - 1;

% Set pixel values accordingly
for i = 1:length(proj_pts)
    if proj_pts(i, 2) <= h && proj_pts(i, 1) <=w && proj_pts(i, 2) >0 && proj_pts(i, 1) >0
        if obj_mask(proj_pts(i,2), proj_pts(i,1)) == -1
           obj_mask(proj_pts(i,2),proj_pts(i,1)) = proj_pts(i,4);
       end    
    end
end

% Make class values from 1 to 8 (0 to 7 before)
obj_mask = obj_mask + 1;

end

