% order 2 polynomial feature mapping
function DK = polyKernelMappingDegree7(D)

    C = 2;
    n = size(D,2); % number of feature dimension
    S1 = D(:,1);
    S2 = D(:,2);
    S3 = D(:,3);
    S4 = D(:,4);
    S5 = D(:,5);
    S6 = D(:,6);
    S7 = D(:,7);    
    
    DK = zeros(size(D,1), nchoosek(n+2, 2));
    
    DK(:,1) = C;
    DK(:,2) = sqrt(2*C)*S1;
    DK(:,3) = S1.^2;
    DK(:,4) = sqrt(2*C)*S2;
    DK(:,5) = sqrt(2)*S1.*S2;
    DK(:,6) = S2.^2;
    DK(:,7) = sqrt(2*C)*S3;
    DK(:,8) = sqrt(2)*S1.*S3;
    DK(:,9) = sqrt(2)*(S2.*S3);
    DK(:,10) = S3.^2;
    DK(:,11) = sqrt(2*C)*S4;
    DK(:,12) = sqrt(2)*(S1.*S4);
    DK(:,13) = sqrt(2)*(S2.*S4);
    DK(:,14) = sqrt(2)*(S3.*S4);
    DK(:,15) = S4.^2;
    DK(:,16) = sqrt(2*C)*S5;
    DK(:,17) = sqrt(2)*(S1.*S5);
    DK(:,18) = sqrt(2)*(S2.*S5);
    DK(:,19) = sqrt(2)*(S3.*S5);
    DK(:,20) = sqrt(2)*(S4.*S5);
    DK(:,21) = S5.^2;    
    DK(:,22) = sqrt(2*C)*S6;
    DK(:,23) = sqrt(2)*(S1.*S6);
    DK(:,24) = sqrt(2)*(S2.*S6);
    DK(:,25) = sqrt(2)*(S3.*S6);
    DK(:,26) = sqrt(2)*(S4.*S6);
    DK(:,27) = sqrt(2)*(S5.*S6);
    DK(:,28) = S6.^2;    
    DK(:,29) = sqrt(2*C)*S7;
    DK(:,30) = sqrt(2)*(S1.*S7);
    DK(:,31) = sqrt(2)*(S2.*S7);
    DK(:,32) = sqrt(2)*(S3.*S7);
    DK(:,33) = sqrt(2)*(S4.*S7);
    DK(:,34) = sqrt(2)*(S5.*S7);
    DK(:,35) = sqrt(2)*(S6.*S7);
    DK(:,36) = S7.^2;        
    
end