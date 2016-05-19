#Smooth_Closed_Surface_Quadrature_RBF_Test()
using PyCall
using NearestNeighbors
@pyimport scipy.spatial.qhull as spsq
include("Smooth_Closed_Surface_Quadrature_RBF.jl")
#============================================================================#
#                                                                            #
# This function provides example calls to the function                       #
# Smooth_Closed_Surface_Quadrature_RBF.jl.  The user can change the parameter#
# Number_of_Quadrature_Nodes in this jl-file to generate various sizes of    #
# node sets.  The user can also change the parameters Poly_Order and         #
# Number_of_Nearest_Neighbors within the                                     #
# Smooth_Closed_Surface_Quadrature_RBF.jl function.                          #
#                                                                            #
# This function approximates the surface integral of five test integrands    #
# over the surface of the sphere and prints the error in the approximation   #
# to the command windown upon completion.  The sphere is used here since it  #
# is a simple matter to generate random node sets for the surface.           #
#                                                                            #
# The function Smooth_Closed_Surface_Quadrature_RBF.jl is the default        #
# implemenation of the method described in:                                  #
#                                                                            #
# J. A. Reeger, B. Fornberg, and M. L. Watts "Numerical quadrature over      #
# smooth, closed surfaces".                                                  #
#                                                                            #
#============================================================================#

#============================================================================#
#                                                                            #
# Generate quadrature nodes on the surface of a sphere of radius 1 using     #
# the method describe in A. Gonzalez, "Measurement of Areas on a Sphere      #
# using Fibonacci and Latitude-Longitude Lattices", Math. Geosci., vol. 42,  #
# number 1, pages 49-64, 2010                                                #
#                                                                            #
#============================================================================#

# Parameters user can set
Number_of_Quadrature_Nodes=2500;

# Generate the quadrature nodes
phi=(1+sqrt(5))/2;

Quadrature_Nodes=zeros(Number_of_Quadrature_Nodes+1,3);

N=Number_of_Quadrature_Nodes/2;

Sphere_Radius=1;

for Index=-N:N
    lat=asin((2*Index)/(2*N+1));
    lon=mod(Index,phi)*2*pi/phi;
    if lon<-pi
        lon=2*pi+lon;
    end
    if lon>pi
        lon=lon-2*pi;
    end
    Quadrature_Nodes[Index+N+1,:]=[cos(lon)*cos(lat) sin(lon)*cos(lat) sin(lat)];
end

# Generate a triangulation of the surface
hull=spsq.ConvexHull(Quadrature_Nodes);
Triangles=hull[:simplices];
Triangles=float64(Triangles+1);
#==========================================================================#

#============================================================================#
#                                                                            #
# Define the level surface h(x,y,z)=0 and its gradient so that the normal    #
# to the surface can be computed exactly.  These are inputs to the function  #
# that can be omitted.  They have the following specifications               #
#                                                                            #
#   h (optional) - For the surface S defined implicitly by h(x,y,z)=0, row i #
#   in the output of h should contain                                        #
#   h(Quadrature_Nodes(i,1:3))                                               #
#   h should take in Quadrature_Nodes as an                                  #
#   (Number_of_Quadrature_Nodes X 3) Array                                   #
#                                                                            #
#   gradh (optional) - The gradient of the function h.  Row i in the output  #
#   of gradh should contain                                                  #
#   [dh/dx(Quadrature_Nodes(i,1:3)),dh/dy(Quadrature_Nodes(i,1:3)),dh/dz(Quadrature_Nodes(i,1:3)]#
#   gradh should take in Quadrature_Nodes as an                              #
#   (Number_of_Quadrature_Nodes X 3) Array                                   #
#                                                                            #
#============================================================================#
function h(p)
    hout=p[:,1].*p[:,1]+p[:,2].*p[:,2]+p[:,3].*p[:,3]-Sphere_Radius*Sphere_Radius;
    return hout
end
function gradh(p)
    gradhout=2*p;
    return gradhout
end
#============================================================================#

#============================================================================#
#                                                                            #
# Generate the quadrature weights for the set of quadrature nodes generated  #
# above.                                                                     #
#                                                                            #
#============================================================================#
# Use the exact surface normal
Quadrature_Weights_Exact_Normal=Smooth_Closed_Surface_Quadrature_RBF(Quadrature_Nodes,Triangles,h,gradh);

# Use the approximate surface normal
Quadrature_Weights_Approx_Normal=Smooth_Closed_Surface_Quadrature_RBF(Quadrature_Nodes,Triangles);
#============================================================================#

#============================================================================#
#                                                                            #
# Define test integrands                                                     #
#                                                                            #
#============================================================================#
function f1(x,y,z)
    f=1+x+y.^2+x.^2.*y+x.^4+y.^5+x.^2.*y.^2.*z.^2;
    return f
end

function f2(x,y,z)
    f=(0.75.*exp(-(9.*x-2).^2./4-(9.*y-2).^2./4-(9.*z-2).^2./4)+
        0.75.*exp(-(9.*x+1).^2./49-(9.*y+1)./10-(9.*z+1)./10)+
        0.5.*exp(-(9.*x-7).^2./4-(9.*y-3).^2./4-(9.*z-5).^2./4)-
        0.2.*exp(-(9.*x-4).^2-(9.*y-7).^2-(9.*z-5).^2));
    return f
end

function f3(x,y,z)
    f=(1+tanh(-9.*x-9.*y+9.*z))./9;
    return f
end

function f4(x,y,z)
    f=(1+sign(-9.*x-9.*y+9.*z))./9;
    return f
end

function f5(x,y,z)
    f=ones(size(x));
    return f
end
#============================================================================#

#============================================================================#
#                                                                            #
# Compute the values of various test integrands at the quadrature nodes      #
# generated above.                                                           #
#                                                                            #
#============================================================================#
F1=f1(Quadrature_Nodes[:,1],Quadrature_Nodes[:,2],Quadrature_Nodes[:,3]);
F2=f2(Quadrature_Nodes[:,1],Quadrature_Nodes[:,2],Quadrature_Nodes[:,3]);
F3=f3(Quadrature_Nodes[:,1],Quadrature_Nodes[:,2],Quadrature_Nodes[:,3]);
F4=f4(Quadrature_Nodes[:,1],Quadrature_Nodes[:,2],Quadrature_Nodes[:,3]);
F5=f5(Quadrature_Nodes[:,1],Quadrature_Nodes[:,2],Quadrature_Nodes[:,3]);
#============================================================================#

#============================================================================#
#                                                                            #
# Set the exact values of the surface integrals of the test integrands for   #
# comparison.                                                                #
#                                                                            #
#============================================================================#
Exact_Surface_Integral_f1=216*pi/35;
Exact_Surface_Integral_f2=6.6961822200736179523;
Exact_Surface_Integral_f3=4*pi/9;
Exact_Surface_Integral_f4=4*pi/9;
Exact_Surface_Integral_f5=4*pi;
#============================================================================#

#============================================================================#
#                                                                            #
# Compute the approximate values of the surface integrals of the test        #
# integrands using the quadrature weights generated above for comparison.    #
#                                                                            #
#============================================================================#
Approximate_Surface_Integral_f1_Exact_Normal=F1.'*Quadrature_Weights_Exact_Normal;
Approximate_Surface_Integral_f2_Exact_Normal=F2.'*Quadrature_Weights_Exact_Normal;
Approximate_Surface_Integral_f3_Exact_Normal=F3.'*Quadrature_Weights_Exact_Normal;
Approximate_Surface_Integral_f4_Exact_Normal=F4.'*Quadrature_Weights_Exact_Normal;
Approximate_Surface_Integral_f5_Exact_Normal=F5.'*Quadrature_Weights_Exact_Normal;

Approximate_Surface_Integral_f1_Approx_Normal=F1.'*Quadrature_Weights_Approx_Normal;
Approximate_Surface_Integral_f2_Approx_Normal=F2.'*Quadrature_Weights_Approx_Normal;
Approximate_Surface_Integral_f3_Approx_Normal=F3.'*Quadrature_Weights_Approx_Normal;
Approximate_Surface_Integral_f4_Approx_Normal=F4.'*Quadrature_Weights_Approx_Normal;
Approximate_Surface_Integral_f5_Approx_Normal=F5.'*Quadrature_Weights_Approx_Normal;
#============================================================================#

#============================================================================#
#                                                                            #
# Compute the error in the approximation of the surface integrals for the    #
# test integrands.                                                           #
#                                                                            #
#============================================================================#
Error_in_the_Approximate_Surface_Integral_f1_Exact_Normal=abs(Exact_Surface_Integral_f1-
    Approximate_Surface_Integral_f1_Exact_Normal)/abs(Approximate_Surface_Integral_f1_Exact_Normal);
Error_in_the_Approximate_Surface_Integral_f2_Exact_Normal=abs(Exact_Surface_Integral_f2-
    Approximate_Surface_Integral_f2_Exact_Normal)/abs(Approximate_Surface_Integral_f2_Exact_Normal);
Error_in_the_Approximate_Surface_Integral_f3_Exact_Normal=abs(Exact_Surface_Integral_f3-
    Approximate_Surface_Integral_f3_Exact_Normal)/abs(Approximate_Surface_Integral_f3_Exact_Normal);
Error_in_the_Approximate_Surface_Integral_f4_Exact_Normal=abs(Exact_Surface_Integral_f4-
    Approximate_Surface_Integral_f4_Exact_Normal)/abs(Approximate_Surface_Integral_f4_Exact_Normal);
Error_in_the_Approximate_Surface_Integral_f5_Exact_Normal=abs(Exact_Surface_Integral_f5-
    Approximate_Surface_Integral_f5_Exact_Normal)/abs(Approximate_Surface_Integral_f5_Exact_Normal);

Error_in_the_Approximate_Surface_Integral_f1_Approx_Normal=abs(Exact_Surface_Integral_f1-
    Approximate_Surface_Integral_f1_Approx_Normal)/abs(Approximate_Surface_Integral_f1_Approx_Normal);
Error_in_the_Approximate_Surface_Integral_f2_Approx_Normal=abs(Exact_Surface_Integral_f2-
    Approximate_Surface_Integral_f2_Approx_Normal)/abs(Approximate_Surface_Integral_f2_Approx_Normal);
Error_in_the_Approximate_Surface_Integral_f3_Approx_Normal=abs(Exact_Surface_Integral_f3-
    Approximate_Surface_Integral_f3_Approx_Normal)/abs(Approximate_Surface_Integral_f3_Approx_Normal);
Error_in_the_Approximate_Surface_Integral_f4_Approx_Normal=abs(Exact_Surface_Integral_f4-
    Approximate_Surface_Integral_f4_Approx_Normal)/abs(Approximate_Surface_Integral_f4_Approx_Normal);
Error_in_the_Approximate_Surface_Integral_f5_Approx_Normal=abs(Exact_Surface_Integral_f5-
    Approximate_Surface_Integral_f5_Approx_Normal)/abs(Approximate_Surface_Integral_f5_Approx_Normal);
#============================================================================#

#============================================================================#
#                                                                            #
# Print some stuff                                                           #
#                                                                            #
#============================================================================#
print("====================================================================\n")
print("The relative error in the approximation of the surface integral of\n")
print("f1(x,y,z)= \n")
print("         4    2  2  2    2          5    2\n")
print("        x  + x  y  z  + x  y + x + y  + y  + 1\n")
print("over the sphere surface (radius 1) with exact normal is\n")
print(Error_in_the_Approximate_Surface_Integral_f1_Exact_Normal[1])
print("\n")
print("====================================================================\n")
print("The relative error in the approximation of the surface integral of\n")
print("f2(x,y,z)=                                                        \n")
print("          /                        2     \ \n")
print("     3    |   9 y   9 z   (9 x + 1)    1 |\n")
print("     - exp| - --- - --- - ---------- - - |\n")
print("     4    \    10    10       49       5 /\n")
print("\n")
print("     1               2            2            2\n")
print("   - - exp(- (9 x - 4)  - (9 y - 7)  - (9 z - 5) )\n")
print("     5 \n")
print("\n")
print("          /            2            2            2 \ \n")
print("     3    |   (9 x - 2)    (9 y - 2)    (9 z - 2)  |\n")
print("   + - exp| - ---------- - ---------- - ---------- |\n")
print("     4    \        4            4            4     /\n")
print("\n")
print("           /            2            2            2 \ \n")
print("     1     |   (9 x - 7)    (9 y - 3)    (9 z - 5)  |\n")
print("   + -  exp| - ---------- - ---------- - ---------- |\n")
print("     2     \        4            4            4     /\n")
print("over the sphere surface (radius 1) with exact normal is\n")
print(Error_in_the_Approximate_Surface_Integral_f2_Exact_Normal[1])
print("\n")
print("====================================================================\n")
print("The relative error in the approximation of the surface integral of\n")
print("f3(x,y,z)=                                                        \n")
print("     1   tanh(9 x + 9 y - 9 z)\n")
print("   - -   ---------------------\n")
print("     9             9          \n")
print("over the sphere surface (radius 1) with exact normal is\n")
print(Error_in_the_Approximate_Surface_Integral_f3_Exact_Normal[1])
print("\n")
print("====================================================================\n")
print("The relative error in the approximation of the surface integral of\n")
print("f4(x,y,z)=                                                        \n")
print("     1   sign(9 x + 9 y - 9 z)\n")
print("   - -   ---------------------\n")
print("     9             9          \n")
print("over the sphere surface (radius 1) with exact normal is\n")
print(Error_in_the_Approximate_Surface_Integral_f4_Exact_Normal[1])
print("\n")
print("====================================================================\n")
print("The relative error in the approximation of the surface integral of\n")
print("f5(x,y,z)=1 over the sphere surface (radius 1) with exact normal is\n")
print(Error_in_the_Approximate_Surface_Integral_f5_Exact_Normal[1])
print("\n")
print("====================================================================\n")
print("\n")
print("\n")
print("\n")
print("====================================================================\n")
print("The relative error in the approximation of the surface integral of\n")
print("f1(x,y,z)= \n")
print("         4    2  2  2    2          5    2\n")
print("        x  + x  y  z  + x  y + x + y  + y  + 1\n")
print("over the sphere surface (radius 1) with approximate normal is\n")
print(Error_in_the_Approximate_Surface_Integral_f1_Approx_Normal[1])
print("\n")
print("====================================================================\n")
print("The relative error in the approximation of the surface integral of\n")
print("f2(x,y,z)=                                                        \n")
print("          /                        2     \ \n")
print("     3    |   9 y   9 z   (9 x + 1)    1 |\n")
print("     - exp| - --- - --- - ---------- - - |\n")
print("     4    \    10    10       49       5 /\n")
print("\n")
print("     1               2            2            2\n")
print("   - - exp(- (9 x - 4)  - (9 y - 7)  - (9 z - 5) )\n")
print("     5 \n")
print("\n")
print("          /            2            2            2 \ \n")
print("     3    |   (9 x - 2)    (9 y - 2)    (9 z - 2)  |\n")
print("   + - exp| - ---------- - ---------- - ---------- |\n")
print("     4    \        4            4            4     /\n")
print("\n")
print("           /            2            2            2 \ \n")
print("     1     |   (9 x - 7)    (9 y - 3)    (9 z - 5)  |\n")
print("   + -  exp| - ---------- - ---------- - ---------- |\n")
print("     2     \        4            4            4     /\n")
print("over the sphere surface (radius 1) with approximate normal is\n")
print(Error_in_the_Approximate_Surface_Integral_f2_Approx_Normal[1])
print("\n")
print("====================================================================\n")
print("The relative error in the approximation of the surface integral of\n")
print("f3(x,y,z)=                                                        \n")
print("     1   tanh(9 x + 9 y - 9 z)\n")
print("   - -   ---------------------\n")
print("     9             9          \n")
print("over the sphere surface (radius 1) with approximate normal is\n")
print(Error_in_the_Approximate_Surface_Integral_f3_Approx_Normal[1])
print("\n")
print("====================================================================\n")
print("The relative error in the approximation of the surface integral of\n")
print("f4(x,y,z)=                                                        \n")
print("     1   sign(9 x + 9 y - 9 z)\n")
print("   - -   ---------------------\n")
print("     9             9          \n")
print("over the sphere surface (radius 1) with approximate normal is\n")
print(Error_in_the_Approximate_Surface_Integral_f4_Approx_Normal[1])
print("\n")
print("====================================================================\n")
print("The relative error in the approximation of the surface integral of\n")
print("f5(x,y,z)=1 over the sphere surface (radius 1) with approximate normal is\n")
print(Error_in_the_Approximate_Surface_Integral_f5_Approx_Normal[1])
print("\n")
print("====================================================================\n")
print("\n")
#============================================================================#
