%% Load .mat file
load('C:\Users\Teo\Documents\Engineering\Year4\4YP\Data\Masks\car_pascal\single\2008_000163_mask_1.mat','single_mask')
opt = globals();

h = length(single_mask(:,1));
w = length(single_mask(1,:));

colors = load(fullfile(opt.path_pascal3d, '/CAD/colors.mat'));
colors = colors.colors;

%% Create single mask image for vizualization
pic_single = zeros(h,w,3);
for variety = 1:8
    for height = 1:h
        for width = 1:w
            if single_mask(height,width) == variety
                pic_single(height, width, :) = colors(variety, :);
            end
        end
    end
end


%% Show the single mask
subplot(1,1,1);
imshow(pic_single);
axis off;
axis equal;