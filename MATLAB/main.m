% Show Pascal3D+ Annotations
function main

cls = 'car'; % Specify the class
opt = globals();
visualizations = 0; % 1-ON, 0-OFF

%% Load PASCAL3D+ Modified CAD Models
fprintf('Load CAD Models from File\n');
file = fullfile(opt.path_pascal3d, '/CAD/labelled_cads_centre.mat');
file = load(file);
cads = file.cads;

%% Load Annotations
filename = fullfile(sprintf(opt.path_ann_imagenet, cls), '*.mat');
files = dir(filename);

%% Get Number of Images
nimages = length(files);


%% Loop through the images
if visualizations
    figure(1);
end
cmap = colormap(jet);
colors = load(fullfile(opt.path_pascal3d, '/CAD/colors.mat'));
colors = colors.colors;
for img_idx = 1:nimages
    filename = files(img_idx).name;
    [pathstr, name, ext] = fileparts(filename);
    
    fprintf('%d %s\n', img_idx, filename);
    im_name = filename;
    
    %% Show Image
    filename = fullfile(sprintf(opt.path_img_imagenet, cls), [name '.jpeg']);
    I = imread(filename);
    [h, w, ~] = size(I);
    if visualizations
        subplot(1, 2, 1);
        imshow(I);
        hold on;
    end
    
    %% Load Annotations
    filename = fullfile(sprintf(opt.path_ann_imagenet, cls), files(img_idx).name);
    object = load(filename);
    objects = object.record.objects;
    
    %% For All Annotated Objects Do
    obj_mask = zeros(h,w);
    depth_map = zeros(h,w);
    for i = 1:numel(objects)
        object = objects(i);
        if strcmp(object.class, cls) == 0
            %disp('Classes do not match!')
            continue;
        end
        
        %% Plot 2D Bounding Box
        bbox = object.bbox;
        bbox_draw = [bbox(1) bbox(2) bbox(3)-bbox(1) bbox(4)-bbox(2)];
        if visualizations
            rectangle('Position', bbox_draw, 'EdgeColor', 'g');
        end
        
        %% Get Vertices and Faces
        cad_index = object.cad_index;
        x2d = project_3d(cads(cad_index), object);
        is_empty = 0;
        if isempty(x2d)
            disp('x2d is empty!');
            is_empty = 1;
            continue;
        end
        faces = cads(cad_index).faces;
        classes = cads(cad_index).vertices(:,4);
        
        %% Draw The CAD Overlap
        index_color = 1 + floor((i-1) * size(cmap,1) / numel(objects));
        if visualizations
            a = patch('vertices', x2d(:,1:2), 'faces', faces, ...
                'FaceColor', cmap(index_color,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
        
        %% Determine Segmentation Mask
        [obj_mask, depth_map] = segmentation_mask(x2d, classes, faces, h, w, obj_mask, depth_map);
    end
    if visualizations
        hold off;
        
        %% Create mask image for vizualization
        pic = zeros(h,w,3);
        for variety = 1:8
            for height = 1:h
                for width = 1:w
                    if obj_mask(height,width) == variety
                        pic(height, width, :) = colors(variety, :);
                    end
                end
            end
        end
        
        
        %% Show the mask
        subplot(1,2,2);
        imshow(pic);
        axis off;
        axis equal;
    end
    
    %% Save the generated segmentation mask
    if ~is_empty
        folder = fullfile(opt.path_pascal3d, '/Masks/car_imagenet', strcat(im_name(1:end-4),'_mask.csv'));
        writematrix(obj_mask, folder);
    end
end

end