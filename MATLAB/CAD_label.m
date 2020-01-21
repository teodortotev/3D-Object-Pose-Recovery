function CAD_label

%% Initialize
opt = globals();            % Set data directories
cls = 'car';                % Specify the class of interest
normal = eye(3);            % Normal vectors to xy, xz, yz
c = [0 0 0];


%% Load PASCAL3D+ CAD Models
disp('Load CAD Models from File');
filename = sprintf(opt.path_cad, cls);
object = load(filename);
cads = object.(cls);

%% Loop Through The CAD Models
for i = 1:length(cads)
    x3d = cads(i).vertices;
    x3d(:,4:6) = zeros;
    
    %% Find The Centroid of Each Model
    %for dim = 1:3
    %    c(i, dim) = sum(x3d(:,dim))/length(x3d(:,dim));
    %end
    
    %% Label Each Point by Disecting the Model with 3 Centroid Planes
    for point = 1:length(x3d)
        for plane = 1:3
            vec = x3d(point, 1:3) - c(1:3);        % Modify if you want to use the centroid
            if (dot(vec, normal(plane, :))) > 0
                x3d(point, 3+plane) = 1;
            end
        end
        
        % Convert Binary Labels to 1 of 8 classes (1 to 8)
        x3d(point, 4) = x3d(point, 4) + 2*x3d(point, 5) + 4*x3d(point, 6) + 1;
    end
    
    % Assign the class labels to the CAD models
    cads(i).vertices = x3d(:,1:4);

end

% Save the modified CAD models as mat files
save("labelled_cads_centre.mat","cads");

end