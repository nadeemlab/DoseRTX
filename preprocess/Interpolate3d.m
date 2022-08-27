
cases = dir('/data/MSKCC-Intern-2021/Dose-Echo-Data/dose_beamlet_3D_martices/influenceMatrix_beamlet_info/');
for k = 3 : length(cases)
  fprintf('Case #%d = %s\n', k-2, cases(k).name);
  filename = strcat('/data/MSKCC-Intern-2021/Dose-Echo-Data/dose_beamlet_3D_martices/influenceMatrix_beamlet_info/', cases(k).name, '/', 'Dose.txt');
  gridPoints = dlmread(filename);

  filename = strcat('/data/MSKCC-Intern-2021/Dose-Echo-Data/dose_beamlet_3D_martices/influenceMatrix_beamlet_info/', cases(k).name, '/', 'grid_res_manual.txt');
  grid_res = dlmread(filename);
  xRes = grid_res(1);
  yRes = grid_res(2);
  zRes = grid_res(3);

  xRange = min(gridPoints(:,1)):xRes:max(gridPoints(:,1));
  yRange = min(gridPoints(:,2)):yRes:max(gridPoints(:,2));
  zRange = min(gridPoints(:,3)):zRes:max(gridPoints(:,3));
  [Xq,Yq,Zq] = meshgrid(xRange, yRange, zRange);
  %Vq = interp3(matrix(:,1),matrix(:,2),matrix(:,3),matrix(:,4),Xq,Yq,Zq);
  vq = griddata(gridPoints(:,1),gridPoints(:,2),gridPoints(:,3),gridPoints(:,4),Xq,Yq,Zq);
  %vq is dose at mesh user resolution

  vq(isnan(vq)) = 0;
  %scatter3(Xq(:), Yq(:), Zq(:),321, vq(:))
  out = [Xq(:) Yq(:) Zq(:) vq(:)];

  filename = strcat('/data/MSKCC-Intern-2021/Dose-Echo-Data/dose_beamlet_3D_martices/influenceMatrix_beamlet_info/', cases(k).name, '/', 'Dose2.txt');
  dlmwrite(filename,out, 'delimiter',' ');
end
