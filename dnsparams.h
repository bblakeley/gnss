// Define constants for CUDA
#define TX 8
#define TY 8
#define TZ 8

// Temporal Jet, following daSilva 2008, PoF
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
#define n_checkpoint 250		// Number of steps to take between saving full 3D fields for checkpointing
#define n_save2D 100	// Number of timesteps to take between saving 2D slices of field data
#define n_stats 100		// Number of timesteps to take between calculating stats data
#define dt .005 	// Timestep
#define nt 10000		// Total number of timesteps to take in the simulation
#define H (PI/2.0)
#define theta (H/35.0)
#define nu (H/3200.0)
#define Re (1.0/nu)
#define Sc 0.7
#define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
#define k_fil 6.0								// High-pass filter cutoff for initial condition
#define SaveLocation "/home/bblakeley/Documents/Research/GNSS/test/tempjetinv002_4H_2000R2_Re3200_filter/%c.%i"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Isotropic/R2_cuda/%s.2000"
// #define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R2/%s.0"
#define StatsLocation "/home/bblakeley/Documents/Research/GNSS/test/tempjetinv002_4H_2000R2_Re3200_filter/stats/%s"

/*
// Define constants for problem
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
#define n_save2D 100	// Number of timesteps to take between saving 2D slices of field data
#define n_stats 20		// Number of timesteps to take between calculating stats data
#define dt .000817653	// Timestep
#define nt 3000		// Total number of timesteps to take in the simulation
#define Re 100
#define nu (1.0/Re)
#define Sc 0.7
// #define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
#define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define SaveLocation "/home/bblakeley/Documents/Research/GNSS/test/R2/%c.%i"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R2/%s.0"
#define StatsLocation "/home/bblakeley/Documents/Research/GNSS/test/R2/stats/%s"
#define H (PI/3.0)
#define theta (H/35.0)
*/
/*
// Define constants for problem
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
#define n_save 260		// Number of steps to take between saving data
#define dt .0004717653	// Timestep
#define nt 4940		// Total number of timesteps to take in the simulation
#define Re 400
#define Sc 0.7
#define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
// #define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define location "/home/bblakeley/Documents/Research/DNS_Data/Isotropic/Test/R4_cuda_customworksize/%c.%i"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R4/%s.0"
*/
/*
// Define constants for problem
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
#define n_save 260		// Number of steps to take between saving data
#define dt .0002924483	// Timestep
// #define nt 520		// Total number of timesteps to take in the simulation
#define nt 9100		// Total number of timesteps to take in the simulation
#define Re 1600
#define Sc 0.7
// #define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
#define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define SaveLocation "/home/bblakeley/Documents/Research/DNS_Data/Isotropic/Test/R4_cuda_customworksize/%c.%i"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R4/%s.0"
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
#define n_save 100		// Number of steps to take between saving data fields
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
*/
