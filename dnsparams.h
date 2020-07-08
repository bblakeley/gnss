// Define constants for CUDA
#define TX 8
#define TY 8
#define TZ 8
//========================================================================================================
// Neumann Settings
//========================================================================================================

// 256^3 Temporal Jet, following daSilva 2008, PoF
#define NX 256
#define NY 256
#define NZ 256
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define DX (LX/NX)
#define n_checkpoint 2000		// Number of steps to take between saving full 3D fields for checkpointing
#define n_vis 200	// Number of timesteps to take between saving 2D slices of field data
#define n_stats 200		// Number of timesteps to take between calculating stats data
#define dt .008 	// Timestep
#define nt 10000		// Total number of timesteps to take in the simulation
#define H (2.0*PI/6.0)
#define theta (H/35.0)
#define theta_s (H/35.0)
#define nu (1.0/1500.0) ////////////////////////(H/2200.0)
#define Re (1.0/nu)
#define Sc 1.0 //////////////////
#define alpha 1.0 ////////////////////
#define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
//#define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation
#define k_fil 24.0								// High-pass filter cutoff for initial condition
#define rootdir "/home/bblakeley/Documents/Research/GNSS/test/diffusiophoresis/exact_D1500_k64/" //////////
//temporal-jet/inv02_6H_0R2_Re2200_f24_a0033_thetas35/"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R2/%s.0"
#define RAD 1

/*
// 512^3 Temporal Jet, following daSilva 2008, PoF
#define NX 512
#define NY 512
#define NZ 512
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define DX (LX/NX)
#define n_checkpoint 500		// Number of steps to take between saving full 3D fields for checkpointing
#define n_vis 100	// Number of timesteps to take between saving 2D slices of field data
#define n_stats 100		// Number of timesteps to take between calculating stats data
#define dt .002 	// Timestep
#define nt 12000		// Total number of timesteps to take in the simulation
#define H (PI/4.0)
#define theta (H/35.0)
#define nu (H/3200.0)
#define Re (1.0/nu)
#define Sc 0.7
#define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
#define k_fil 96.0								// High-pass filter cutoff for initial condition
#define rootdir "/home/bblakeley/Documents/Research/GNSS/test/temporal-jet/inv2_8H_0R4_Re3200_filter96/"
#define DataLocation "/home/bblakeley/Documents/Research/Flamelet_Data/R4/%s.0"
#define RAD 1
*/
/*
// R2 H.I.T. Simulation Settings
#define NX 256
#define NY 256
#define NZ 256
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define DX (LX/NX)
#define n_checkpoint 150		// Number of steps to take between saving full 3D fields for checkpointing
#define n_vis 20	// Number of timesteps to take between saving 2D slices of field data
#define n_stats 20		// Number of timesteps to take between calculating stats data
#define dt .000817653	// Timestep
#define nt 3000		// Total number of timesteps to take in the simulation
#define Re 100
#define nu (1.0/Re)
#define Sc 0.7
// #define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
#define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define rootdir "/home/bblakeley/Documents/Research/GNSS/test/R2/"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R2/%s.0"
#define H (PI/3.0)
#define theta (H/35.0)
#define k_fil 12.0
#define RAD 1 // Radius of finite difference stencil for area calculations
*/
/*
// R4 H.I.T. Simulation Settings
#define NX 512
#define NY 512
#define NZ 512
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define DX (LX/NX)
#define n_checkpoint 260		// Number of steps to take between saving data
#define n_stats 100
#define dt .0004717653	// Timestep
#define nt 4940		// Total number of timesteps to take in the simulation
#define Re 400
#define nu (1.0/Re)
#define Sc 0.7
#define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
// #define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define rootdir "/home/bblakeley/Documents/Research/DNS_Data/Isotropic/Test/R4_cuda_customworksize/"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R4/%s.0"
#define k_fil 0
#define RAD 1
#define H (PI/4.0)
#define theta (H/35.0)
*/
/*
// R6 H.I.T. Simulation Settings
#define NX 1024
#define NY 1024
#define NZ 1024
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define DX (LX/NX)
#define n_checkpoint 260		// Number of steps to take between saving data
#define dt .0002924483	// Timestep
// #define nt 520		// Total number of timesteps to take in the simulation
#define nt 9100		// Total number of timesteps to take in the simulation
#define Re 1600
#define nu (1.0/Re)
#define Sc 0.7
// #define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
#define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define rootdir "/home/bblakeley/Documents/Research/DNS_Data/Isotropic/Test/R4_cuda_customworksize/"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R4/%s.0"
#define k_fil 0
#define RAD 1
#define H (PI/4.0)
#define theta (H/35.0)
*/
/*
// Taylor-Green Vortex, Re=400
#define NX 256
#define NY 256
#define NZ 256
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define DX (LX/NX)
#define n_checkpoint 100		// Number of steps to take between saving full 3D fields for checkpointing
#define n_vis 20	// Number of timesteps to take between saving 2D slices of field data
#define n_stats 20		// Number of timesteps to take between calculating stats data
#define dt 0.01 	// Timestep
#define nt 1000		// Total number of timesteps to take in the simulation
#define Re 400
#define nu (1.0/Re)
#define Sc 0.7
#define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
// #define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define rootdir "/home/bblakeley/Documents/Research/GNSS/test/taylor-green/n256_re400/"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R2/%s.0"
#define k_fil 0
#define RAD 1
#define H (PI/4.0)
#define theta (H/35.0)
*/

//========================================================================================================
// Digits Settings 
//========================================================================================================
/*
// 256^3 Temporal Jet, following daSilva 2008, PoF
#define NX 256
#define NY 256
#define NZ 256
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define DX (LX/NX)
#define n_checkpoint 500                // Number of steps to take between saving full 3D fields for checkpointing
#define n_vis 500       // Number of timesteps to take between saving 2D slices of field data
#define n_stats 20             // Number of timesteps to take between calculating stats data
#define dt .004         // Timestep
#define nt 100               // Total number of timesteps to take in the simulation
#define H (PI/3.0)
#define theta (H/35.0)
#define nu (H/3200.0)
#define Re (1.0/nu)
#define Sc 0.7
#define k_max (2.0/3.0*(double)NX/2.0)                  // De-alias using the 2/3 truncation rule
#define k_fil 20.0                                                              // High-pass filter cutoff for initial condition
#define RAD 1	// stencil radius for surface area calculations
#define rootdir "/home/bblakeley/Documents/GNSS/test/temporal-jet/inv2_6H_0R2_Re3200_filter48/"
#define DataLocation "/home/bblakeley/Documents/Flamelet_Data/R2/%s.0"
*/
/*
// 512^3 Temporal Jet, following daSilva 2008, PoF
#define NX 512
#define NY 512
#define NZ 512
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define DX (LX/NX)
#define n_checkpoint 500		// Number of steps to take between saving full 3D fields for checkpointing
#define n_vis 100	// Number of timesteps to take between saving 2D slices of field data
#define n_stats 100		// Number of timesteps to take between calculating stats data
#define dt .002 	// Timestep
#define nt 18000		// Total number of timesteps to take in the simulation
#define H (PI/4.0)
#define theta (H/35.0)
#define nu (H/3200.0)
#define Re (1.0/nu)
#define Sc 0.7
#define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
#define k_fil 48.0					// High-pass filter cutoff for initial condition
#define RAD 1
#define rootdir "/home/bblakeley/Documents/GNSS/test/temporal-jet/inv02_8H_0R4_Re3200_filter48/"
#define DataLocation "/home/bblakeley/Documents/Flamelet_Data/R4/%s.0"
*/
/*
// R2 H.I.T. Simulation Settings
#define NX 256
#define NY 256
#define NZ 256
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define DX (LX/NX)
#define n_checkpoint 1000		// Number of steps to take between saving full 3D fields for checkpointing
#define n_vis 100	// Number of timesteps to take between saving 2D slices of field data
#define n_stats 5		// Number of timesteps to take between calculating stats data
#define dt .000817653	// Timestep
#define nt 10		// Total number of timesteps to take in the simulation
#define Re 100
#define nu (1.0/Re)
#define Sc 0.7
// #define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
#define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define rootdir "/home/bblakeley/Documents/GNSS/test/R2/"
#define DataLocation "/home/bblakeley/Documents/Flamelet_Data/R2/%s.0"
#define H (PI/3.0)
#define theta (H/35.0)
#define k_fil 12.0
#define RAD 1 // Radius of finite difference stencil for area calculations
*/
/*
// R4 H.I.T. Simulation Settings
#define NX 512
#define NY 512
#define NZ 512
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define DX (LX/NX)
#define n_checkpoint 260		// Number of steps to take between saving data
#define n_vis 260
#define n_stats 100
#define dt .0004717653	// Timestep
#define nt 4940		// Total number of timesteps to take in the simulation
#define Re 400
#define nu (1.0/Re)
#define Sc 0.7
#define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
// #define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define rootdir "/home/bblakeley/Documents/DNS_Data/Isotropic/Test/R4_cuda_customworksize/"
#define DataLocation "/home/bblakeley/Documents/DNS_Data/Flamelet_Data/R4/%s.0"
#define k_fil 0
#define RAD 1
#define H (PI/4.0)
#define theta (H/35.0)
*/
/*
// R6 H.I.T. Simulation Settings
#define NX 1024
#define NY 1024
#define NZ 1024
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define DX (LX/NX)
#define n_checkpoint 260		// Number of steps to take between saving data
#define n_vis 260
#define n_stats 100
#define dt .0002924483	// Timestep
// #define nt 520		// Total number of timesteps to take in the simulation
#define nt 9100		// Total number of timesteps to take in the simulation
#define Re 1600
#define nu (1.0/Re)
#define Sc 0.7
// #define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
#define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define SaveLocation "/home/bblakeley/Documents/Research/DNS_Data/Isotropic/Test/R4_cuda_customworksize/%c.%i"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R4/%s.0"
#define k_fil 0
#define RAD 1
#define H (PI/4.0)
#define theta (H/35.0)
*/
/*
// Taylor-Green Vortex, Re=400
#define NX 256
#define NY 256
#define NZ 256
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define DX (LX/NX)
#define n_checkpoint 500		// Number of steps to take between saving full 3D fields for checkpointing
#define n_vis 100	// Number of timesteps to take between saving 2D slices of field data
#define n_stats 20		// Number of timesteps to take between calculating stats data
#define dt 0.01 	// Timestep
#define nt 1000		// Total number of timesteps to take in the simulation
#define Re 400
#define nu (1.0/Re)
#define Sc 0.7
#define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
// #define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define SaveLocation "/home/bblakeley/Documents/Research/DNS_Data/GNSS/Test/TG/Re400/%c.%i"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R2/%s.0"
#define StatsLocation "/home/bblakeley/Documents/Research/DNS_Data/GNSS/Test/TG/Re400/stats/%s"
#define k_fil 0
#define RAD 1
#define H (PI/4.0)
#define theta (H/35.0)
*/
