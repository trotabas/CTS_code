#include <vector>
#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <time.h>

#include <Eigen/SparseCore>
#include<Eigen/SparseQR>
#include<Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>
#include<Eigen/IterativeLinearSolvers>
#include <Eigen/Core>
#include <limits>

#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#include <boost/math/tools/roots.hpp>
// #include<boost/range/numeric.hpp>
//==============================================================================================================
//---------------------:NAMES-SPACE-----------------------------------------------------------------------------
//==============================================================================================================
using namespace std;
using dvector = vector<double>;
using namespace Eigen;
namespace broot = boost::math::tools;

//==============================================================================================================
//---------------------:TYPE-DEF--------------------------------------------------------------------------------
//==============================================================================================================
typedef vector< vector<double> > Matrix_type;
typedef numeric_limits< double > dbl;
template <typename TT>
string to_string_with_precision(const TT a_value, const int n = 2)
{
    ostringstream out;
    out.precision(n);
    out << fixed << a_value;
    return out.str();
}

//==============================================================================================================
//---------------------:PROTOTYPE-FUNCTION----------------------------------------------------------------------
//==============================================================================================================
//--> Definition length / surface
void linspace(dvector &linear_vector, double v_start, double v_end, int node);
void definition_Surface(dvector &Sr, dvector &r, double step_r, double re, double rg, int Nr, int Nre);
//--> Print Array
void display_2Darray(Matrix_type& m, string fileName);
void display_VectorXd(VectorXd x, string text);
void display(string text)
{
	cout << text << endl;
}
//--> Output
void saveData(MatrixXd matrix, string fileName,  string path);
bool outputCSV(Matrix_type& matrix_2D, string fileName,  string path);
bool outputCSV_scalar(int scalar, string fileName,  string path);
bool outputCSV_Properties(double mi, double nN, double Omega_e, double Omega_i, double nu_en_0, double nu_ei_0, double nu_in, double step_t);
bool outputCSV_list(VectorXd& list_data, string fileName,  string path);
//->
bool outputCSV_Macho(Matrix_type& matrix_2D, string fileName,  string path);
bool outputCSV_list_Macho(VectorXd& list_data, string fileName,  string path);
void saveData_Macho(MatrixXd matrix, string fileName,  string path, int cols, int rows);
//--> Input
void Input_2Darray(Matrix_type& m, string fileName);
//--> Read input file
map<string, double> loadParameters(const char *  inputFile);
//--> Initialization matrix
void initialization_density(Matrix_type& ne, double ne_background, dvector &ne_init, string fileName);
void initialization_potential_Vr(VectorXd& phi, VectorXd &phi_init, string fileName);
//--> Copy 2D array
void copy_MatrixType(Matrix_type& matrix_2D_save, Matrix_type& matrix_2D);
void copy_VectorXdType(VectorXd& phi_save, VectorXd& phi);
//==============================================================================================================
//---------------------:FUNCTION----------------------------------------------------------------------
//==============================================================================================================
double find_electrode_bias(double x, double I_k, VectorXd j_sh, VectorXd phi_sh, VectorXd Te_sh, double j_eth, double Lambda, int dim)  
{
    double sum = 0.;

    for (int j=0; j<dim; j++)
    {
        sum += j_sh[j]*(1 + (j_eth/j_sh[j]) - exp(Lambda + ((x - phi_sh[j])/Te_sh[j])));
    }
    return sum-I_k;
} 



int main(int argc, char *args[])
{
	//===================================================================================================
	//---------------------:INSERT-NAME-FOLDER-----------------------------------------------------------
	//===================================================================================================
    string folderName 	      = "Test06";
	//===================================================================================================
	//---------------------------------------------------------------------------------------------------
	//===================================================================================================
	clock_t t_all = clock();
	cout.precision(dbl::max_digits10);
	map<string, double> parametersMap;
	parametersMap = loadParameters("input.dat");
	//===================================================================================================
	//---------------------:OPTION-----------------------------------------------------------------------
	//===================================================================================================
	int magnetic_effect        = parametersMap["magnetic_effect"];          //------Option---01----------
    int inertial_effect        = parametersMap["inertial_effect"];          //------Option---02----------
    int Source_term_uniform    = parametersMap["Source_term_uniform"];      //------Option---03----------
    int Thermionic_emission    = parametersMap["Thermionic_emission"];      //------Option---04----------
    int Richardson_law         = parametersMap["Richardson_law"];           //------Option---05----------
    int evo_electrode_bias     = parametersMap["evo_electrode_bias"];       //------Option---06----------
    int break_time             = parametersMap["break_time"];               //------Option---07----------
    int only_inside_CD         = parametersMap["only_inside_CD"];           //------Option---08----------     
    int Eq_continuity          = parametersMap["Eq_continuity"];            //------Option---08----------   
	//===================================================================================================
	//---------------------:OUTPUT-FOLDER-PATH-----------------------------------------------------------
	//===================================================================================================    
	string path_iteration  = "Output/" + folderName  + "/Iteration";
	string path_final      = "Output/" + folderName  + "/Final";
	cout << folderName << endl;
    const char * path = path_iteration.c_str();
    // checking if folder already exist
    struct stat buffer;

    if (stat(path, &buffer) != 0)
    {
        // Creating a directory
        if (mkdir(path, 0777) == -1)
            cerr << "Error :  " << strerror(errno) << endl;

        else
            cout << "Directory : " + path_iteration + " created" << endl;

    }
    else
    {
    	cout << "Directory : " + path_iteration  + " exists" << endl;
    }
	
    const char * path_Macho = path_final.c_str();
    // checking if folder already exist

    if (stat(path_Macho, &buffer) != 0)
    {
        // Creating a directory
        if (mkdir(path_Macho, 0777) == -1)
            cerr << "Error :  " << strerror(errno) << endl;

        else
            cout << "Directory : " + path_final +" created" << endl;

    }
    else
    {
    	cout << "Directory : " + path_final +" exists" << endl;
    }
	//===================================================================================================
	//---------------------:PHYSICAL-CONSTANT------------------------------------------------------------
	//===================================================================================================
	double NA           = parametersMap["NA"];              //-------Avogadro----Number---------[mole^-1]
	double kB           = parametersMap["kB"];              //-------Boltzmann---Constant-------[Nr^2.kg.s^-2.K^-1]
	double CoulombLog   = parametersMap["CoulombLog"];      //-------Coulomnb----Logarithm------
	double eps0         = parametersMap["eps0"];            //-------Vacuum------Permittivity---[Nr^-3.kg^-1.s^4.A^2]
	double sig0         = parametersMap["sig0"];	        //-------Cross-------Section--------[Nr-2]
	double e            = parametersMap["e"];               //-------Eletric-----Charge---------[C]
	double me           = parametersMap["me"];              //-------Electron----Mass-----------[kg]
	double AG           = parametersMap["AG"];              //-------Richardson--Constant-------[A.K^-2.Nr^-2]
	//===================================================================================================
	//---------------------:PLASMA-PARAMETERS------------------------------------------------------------
	//===================================================================================================
	double mi_amu       = parametersMap["mi_amu"];          //-------Atomic----Mass-------------[amu]
	double mi           = mi_amu*1.6605e-27;		        //-------Ion-------Mass-------------[kg]
	double Ti           = parametersMap["Ti"];              //-------Ion-------Temperature------[eV]
	double Tn           = parametersMap["Tn"];              //-------Room------Temperature------[K]
	double B            = parametersMap["B"];               //-------Magnetic--Field------------[T]
	double P            = parametersMap["P"];               //-------Neutral---Pressure---------[Pa]
	double nN           = (1/(kB*Tn))*P;		            //-------Neutral---Density----------[Nr^-3]
	double T0           = parametersMap["T0"];              //-------Electron--Temperature------[eV]
	double n0           = parametersMap["n0"];	            //-------Density--------------------[m^-3]
	//===================================================================================================
	//---------------------:GEOMETRICAL-PARAMETERS-------------------------------------------------------
	//===================================================================================================
	double rg           = parametersMap["rg"];              //------Plasma--column--radius------[Nr]
	double L            = parametersMap["L"];               //------Plasma--column--length------[Nr]
	//===================================================================================================
	//---------------------:TIME-SETTING-----------------------------------------------------------------
	//===================================================================================================
	double step_t       = parametersMap["step_t"];          //------Time-------Step-------------
	int iteration_save  = parametersMap["iteration_save"];  //------Save---data--every----------
	int iteration_print = parametersMap["iteration_print"]; //------Print--data--every----------
	//===================================================================================================
	//---------------------:MESHING-PARAMETERS-----------------------------------------------------------
	//===================================================================================================
	int Nr              = parametersMap["Nr"];              //------Radial-----Node-------------
	int Nre             = parametersMap["Nre"];             //------Radial-----Node-------------
	double step_r       = rg/(Nr-1);			            //------Radial-----Step-------------
	dvector r(Nr);
	linspace(r, 0, rg, Nr);
    double re = r[Nre-1];
    cout << "re = " << re << endl;
    dvector Sr(Nr);
    definition_Surface(Sr, r, step_r, re, rg, Nr, Nre);
	//-->
	int Nz              = parametersMap["Nz"];              //------Axial------Node-------------
	double step_z       = (L/2.)/(Nz-1);		            //------Axial------Step-------------
	dvector z(Nz);
	linspace(z, -L/2, 0, Nz);
	//-->
	int K               = Nr*Nz;			                //------Total------Node-------------
	//===================================================================================================
    //---------------------:ELECTRODE-POTENTIAL-CURRENT----------------------------------
	//===================================================================================================

	double phi_e        = parametersMap["phi_e"];           //------Electrode--Bias-------------[V]
    double I_k          = parametersMap["I_k"];             //------Target-----Current----------[A]
	//===================================================================================================
    //---------------------:EMISSIVE-ELECTRODE-----------------------------------------------------------
	//===================================================================================================
    double I_eth        = parametersMap["I_eth"];           //------Thermionic-Current----------[A]
	double Tw           = parametersMap["Tw"];              //------Heating----Temperature------[eV]
	double W            = parametersMap["W"];               //------Work-------Function---------[J]
    double j_eth        = 0.;                               //------Thermionic-Current Density--[A.Nr^-2]
    double Sre          = 0.;
    if (Thermionic_emission == 1)
    {
        if (Richardson_law == 1)
        {
            j_eth        =  AG*pow(Tw,2)*exp(-e*W/(kB*Tw));
        }
        else
        {
            for (int j=0; j<Nre-1; j++)
            {
                Sre += Sr[j];
            }
            cout << "pi*re^2 = " << (M_PI*pow(re,2.)) << " and Sre = " << Sre << endl;
            j_eth        = I_eth/Sre;   //(boost::accumulate(*Sr[0], *Sr[Nre-1]));
        }
    }

	//===================================================================================================
	//---------------------:DISTRIBUTION-DENSITY-/-TEMPERATURE-------------------------------------------
	//===================================================================================================
	Matrix_type Te( Nr,vector<double>(Nz, T0) );
	Matrix_type ne( Nr,vector<double>(Nz, n0)  );
	Matrix_type ne_save( Nr,vector<double>(Nz, n0) );
	//===================================================================================================
	//---------------------:TIME-SETTING-----------------------------------------------------------------
	//===================================================================================================
	double Cs_0D  = sqrt(e*Te[0][0]/mi);
	double CFL = Cs_0D*step_t*((1/step_z)+(1/step_r));
	cout << "|CFL| = " << CFL  << endl;
    double max_time_ms = parametersMap["max_time_ms"];
	int max_iteration = parametersMap["max_iteration"];
	//===================================================================================================
	//---------------------:ION-VELOCITY-----------------------------------------------------------------
	//===================================================================================================
	dvector r_v(Nr-1);
	linspace(r_v, step_r/2., rg-step_r/2., Nr-1);
	dvector z_v(Nz-1);
	linspace(z_v, -L/2.+step_z/2., -step_z/2., Nz-1);

	Matrix_type Vr( Nr-1,vector<double>(Nz-1,      0.) );
	Matrix_type Vr_save( Nr-1,vector<double>(Nz-1, 0.) );

	Matrix_type Vtheta( Nr-1,vector<double>(Nz-1,      0.) );
	Matrix_type Vtheta_save( Nr-1,vector<double>(Nz-1, 0.) );

	Matrix_type Vz( Nr-1,vector<double>(Nz-1,      0.) );
	Matrix_type Vz_save( Nr-1,vector<double>(Nz-1, 0.) );

	//===================================================================================================
	//---------------------:CURRENT-DENSITY------------------------------------------------------------------
	//===================================================================================================
    VectorXd j_z_sh_electrode(Nre);
    VectorXd j_z_sh_electrode_BC(Nre);
    VectorXd j_z_sh_drop(Nr-Nre);
    // VectorXd j_z_mid(Nr);
    // VectorXd j_r_axi(Nz);
	//===================================================================================================
	//---------------------:SOURCE-TERM------------------------------------------------------------------
	//===================================================================================================
    double I_src        = parametersMap["I_src"];           //------Source-----Current----------[A]
    double S0           = I_src/(e*M_PI*pow(rg,2.)*(L/2));
    cout << "S0 = " << S0 << endl;
	// double S0           = parametersMap["S0"];              //-------Source----Term-------------[m^-3/s^-1]
	Matrix_type S( Nr,vector<double>(Nz, S0) );
    
    if (Source_term_uniform == 0)
    {
        // Matrix_type S_adm( Nr,vector<double>(Nz, 1) );
        double S_adm = 0.;
        double int_S_V = 0.;
        double Volume = 0.;
        double dz = 0.5;
        double dr = 0.05;
        double z0 = 0.;
        double r0 = 0.;
        for (int i=0; i<Nz; i++)
        {
            for (int j=0; j<Nr; j++)
            {
                if (j == 0)
                {
                    if (i==0 || i == Nz-1)
                    {
                        Volume = M_PI*pow((r[j]+(step_r/2.)),2.)*(step_z/2.);
                    }
                    else
                    {
                        Volume = M_PI*pow((r[j]+(step_r/2.)),2.)*(step_z);
                    }
                }
                else if (j == Nr-1)
                {
                    if (i==0 || i == Nz-1)
                    {
                        Volume = M_PI*(pow(r[j],2.) - pow((r[j]-(step_r/2.)),2.))*(step_z/2.);
                    }
                    else
                    {
                        Volume = M_PI*(pow(r[j],2.) - pow((r[j]-(step_r/2.)),2.))*(step_z);
                    }
                }
                else if (j != 0 && j != Nr-1)
                {
                    if (i==0 || i == Nz-1)
                    {
                        Volume = M_PI*(pow((r[j]+(step_r/2.)),2.) - pow((r[j]-(step_r/2.)),2.))*(step_z/2.);
                    }
                    else
                    {
                        Volume = M_PI*(pow((r[j]+(step_r/2.)),2.) - pow((r[j]-(step_r/2.)),2.))*(step_z);
                    }
                }
                S_adm = exp( -pow((z[i]-z0),2.)/(2*pow(dz,2.)) )*exp( -pow((r[j]-r0),2.)/(2*pow(dr,2.)) );
                int_S_V += S_adm*Volume;
            }
        }
        S0 = I_src/(e*int_S_V);
        for (int i=0; i<Nz; i++)
        {
            for (int j=0; j<Nr; j++)
            {
                S[j][i] = S0*exp( -pow((z[i]-z0),2.)/(2*pow(dz,2.)) )*exp( -pow((r[j]-r0),2.)/(2*pow(dr,2.)) );
            }
        }
    }
    
    outputCSV_Macho(S, "S_BT.csv", path_final);
	//===================================================================================================
	//---------------------:SHEATH-PROPERTIES------------------------------------------------------------
	//===================================================================================================
	double eta    = me/mi;
	double Lambda = log(sqrt(mi/(2*M_PI*me)));
	//===================================================================================================
	//---------------------:Cyclotron frequency----------------------------------------------------------
	//===================================================================================================
	double Omega_i = (e*B)/mi;
	double Omega_e = (e*B)/me;
	//===================================================================================================
	//---------------------:Collision frequency----------------------------------------------------------
	//===================================================================================================
	Matrix_type nu_en( Nr,vector<double>(Nz,0.0) );
	Matrix_type nu_ei( Nr,vector<double>(Nz,0.0) );
	Matrix_type tilde_nu_N( Nr,vector<double>(Nz,0.0) );
	Matrix_type tilde_nu_I( Nr,vector<double>(Nz,0.0) );
	double nu_in = sig0*nN*sqrt((8*e*Ti)/(M_PI*mi));
    cout << "nu_in" << nu_in << endl;
	//===================================================================================================
	//---------------------:Perpendicular Conductivity---------------------------------------------------
	//===================================================================================================
	Matrix_type K0( Nr,vector<double>(Nz,0.0) );
	Matrix_type K1( Nr,vector<double>(Nz,0.0) );
	Matrix_type K2( Nr,vector<double>(Nz,0.0) );
	Matrix_type K3( Nr,vector<double>(Nz,0.0) );
	Matrix_type sigma_perp( Nr,vector<double>(Nz,0.0) );

	//===================================================================================================
	//---------------------:Parallel Conductivity -------------------------------------------------------
	//===================================================================================================
	Matrix_type alpha( Nr,vector<double>(Nz,0.0) );
	Matrix_type sigma_para( Nr,vector<double>(Nz,0.0) );

	//===================================================================================================
	//---------------------:Mobility and Diffusion ------------------------------------------------------
	//===================================================================================================
	Matrix_type mu_i_para( Nr,vector<double>(Nz,0.0) );
	Matrix_type mu_i_perp( Nr,vector<double>(Nz,0.0) );
	Matrix_type mu_e_para( Nr,vector<double>(Nz,0.0) );
	Matrix_type mu_e_perp( Nr,vector<double>(Nz,0.0) );
	Matrix_type D_e_para( Nr,vector<double>(Nz,0.0) );
	Matrix_type D_e_perp( Nr,vector<double>(Nz,0.0) );

    //outputCSV_Properties( mi,  nN,  Omega_e,  Omega_i,  nu_en_0,  nu_ei_0,  nu_in,  step_t);
    


	//===================================================================================================
	//=====================: POTENTIAL SOLVER ===========================================================
	//===================================================================================================
	VectorXd phi(K);
	VectorXd phi_save(K);
	for(int k(0); k<K; k++)
	{
		phi_save[k] = phi_e + Lambda*Te[0][0];
	}
	//Map < MatrixXd, RowMajor> phi_matrix_save( phi_save.data(), Nz,Nr);
	//saveData(phi_matrix_save.transpose(), "phi_save.csv", path_iteration);
    VectorXd j_sh(Nre);
    VectorXd phi_sh(Nre);
    VectorXd Te_sh(Nre);
	//===================================================================================================
	//---------------------:INITIALISATION --------------------------------------------------------------
	//===================================================================================================
	int dim_sheath = 2*Nre+ 2*(Nr-1-Nre) + 1;
	int dim_mid    = 4*(Nr-1) + 1;
	int dim_axisym = 4*(Nz-2);
	int dim_ground = Nz-2;
	int dim_inside = 5*(Nz*Nr-2*Nz-2*(Nr-2));
	int dim_matrix_A = dim_sheath + dim_mid + dim_axisym + dim_ground + dim_inside;
	cout << "dim_A = " << dim_matrix_A << endl;
	///===================================================================================================
	///---------------------:MATRIX A --------------------------------------------------------------------
	///===================================================================================================
	//-->
	double C_ij  = 0.0;
	double C_ipj = 0.0;
	double C_imj = 0.0;
	double C_ijp = 0.0;
	double C_ijm = 0.0;
	//-->
	double mean_01_z_p = 0.0;
	double mean_01_z_m = 0.0;
	double mean_01_r_p = 0.0;
	double mean_01_r_m = 0.0;
	//-->
	double A_z_j  = 0.0;
	double A_r_jp = 0.0;
	double A_r_jm = 0.0;
	int i = 0;
	int j = 0;
	int k = 0;

	double err_L1_phi = 1.0;
	double err_L1_ne  = 1.0;
	double eps = 1e-12;
	int iteration = 0;


	double Nj, dNj = 0.0;
	double F_ip2j = 0.0;
	double F_im2j = 0.0;
	double F_ijp2 = 0.0;
	double F_ijm2 = 0.0;
	double G_ip2j = 0.0;
	double G_im2j = 0.0;
	double G_ijp2 = 0.0;
	double G_ijm2 = 0.0;
	double mean_02_z_p = 0.0;
	double mean_02_z_m = 0.0;
	double mean_02_r_p = 0.0;
	double mean_02_r_m = 0.0;
	double mean_03_z_p = 0.0;
	double mean_03_z_m = 0.0;
	double mean_03_r_p = 0.0;
	double mean_03_r_m = 0.0;
	double I_ip2j = 0.0;
	double I_im2j = 0.0;
	double I_ijp2 = 0.0;
	double I_ijm2 = 0.0;

    double j_is = 0.0;
    double mean_ne_save = 1.;

	//----> Determine the node inside the domaine of "Electrode"
	dvector vect_InsideDomain((Nr-2)*(Nz-2));
	int index_Inside_Domain = 0;
	for (int k = Nz+1; k!=Nz*(Nr-1); k++)
	{
		if (k%Nz!=0 && (k+1)%Nz!=0)
		{
			vect_InsideDomain[index_Inside_Domain] = k;
			index_Inside_Domain++;
		}
	}

    ///---------------------------------------------------> Initialisation
	VectorXd vect_b(K);

    //--------------------------> Error
    double STD_phi_01 = 0.0;
    double STD_phi_02 = 0.0;
    int cpt = 0;
    double STD_ne_01 = 0.0;
    double STD_ne_02 = 0.0;
    double V_ij = 0.0;
    double I_total = 0.;

    ///---------------------------------------------------> Velocity
    double mean_nu_ei_vperp = 0.0;
    double mean_S_ne_vperp  = 0.0;
    double mean_nu_ei_vpara = 0.0;
    double mean_S_ne_vpara  = 0.0;
    double mean_mu_i_perp0  = 0.0;
    double mean_mu_i_para0  = 0.0;
    double nu_I_perp = 0.0;
    double nu_I_para = 0.0;
    //--
    double Vr_dr_Vr = 0.0;
    double Vz_dz_Vr = 0.0;
    //--
    double Vr_dr_Vtheta = 0.0;
    double Vz_dz_Vtheta = 0.0;
    //--
    double Vr_dr_Vz = 0.0;
    double Vz_dz_Vz = 0.0;
    //--
    double Vr_mean = 0.0;
    double Vr_mean_jp = 0.0;
    double Vr_mean_jm = 0.0;
    double Vz_mean = 0.0;
    double Vz_mean_ip = 0.0;
    double Vz_mean_im = 0.0;
    //--m
    double Vz_sh     = -Cs_0D;
    double Vz_mid    = 0.0;
    double Vz_axi    = 0.0; //symetrie : z
    double Vz_ground = 0.0;
    //-->
    double Vr_sh     = 0.0;
    double Vr_mid    = 0.0; //symetrie : r
    double Vr_ground = 0.0; // Cs_0D; // 0.0; //   
    double Vr_axi    = 0.0;
    //-->
    double Vtheta_sh     = 0.0;
    double Vtheta_mid    = 0.0; //symetrie : r
    double Vtheta_ground = 0.0; // //   Cs_0D; // 
    double Vtheta_axi    = 0.0;

    while (err_L1_phi > eps) //  (iteration < 10) //   (iteration < 1) // err_L1_phi // err_L1_ne
	{
		///---------------------------------------------------> Collision Frequency / Conductivity / Mobility / Diffusion
		for (int i=0; i<Nz; i++)
		{
			for (int j=0; j<Nr; j++)
			{
				nu_en[j][i] = sig0*nN*sqrt((8*e*Te[j][i])/(M_PI*me));
				//nu_ei[j][i] = 2.90631687e-12*CoulombLog*ne[j][i]*pow(Te[j][i], -3./2.); // (pow(e,5./2.)*CoulombLog/(6.*sqrt(2.)*pow(M_PI,3./2.)*pow(eps0,2.)*sqrt(me)))*ne[j][i]*pow(Te[j][i], -3./2.);S
				//-->
				mu_e_para[j][i] =  e/(me*(nu_ei[j][i]+nu_en[j][i]));
				D_e_para[j][i]  = Te[j][i]*mu_e_para[j][i];
				mu_i_para[j][i] = e/(mi*(nu_in));

				if (magnetic_effect == 0)
				{
					sigma_perp[j][i] = (pow(e,2)*ne[j][i]/(mi*nu_in + me*nu_ei[j][i]))*(1 + (1/(alpha[j][i] + nu_en[j][i]/nu_ei[j][i]))*(((mi*nu_in)/(me*nu_ei[j][i])) - alpha[j][i]) );
					//-->
					mu_e_perp[j][i] = e/(me*(nu_ei[j][i]+nu_en[j][i]));
					mu_i_perp[j][i] = e/(mi*(nu_in));
					D_e_perp[j][i]  = Te[j][i]*mu_e_perp[j][i];

				}
				else if (magnetic_effect == 1)
				{
					//--> Mobility
					mu_e_perp[j][i] = (me/(e*pow(B,2.)))*((nu_ei[j][i]+nu_en[j][i])/(1 + pow((nu_ei[j][i]+nu_en[j][i]),2)/pow(Omega_e,2)));
					mu_i_perp[j][i] = (mi/(e*pow(B,2.)))*((nu_in)/(1 + pow((nu_in),2)/pow(Omega_i,2)));
					//--> Diffusion
					D_e_perp[j][i]  = Te[j][i]*mu_e_perp[j][i];
				}
			}
		}

        ///---------------------------------------------------> Initialisation
        typedef Triplet<double,int> T;
        vector<T> entries;
        entries.reserve(dim_matrix_A);


        ///---------------------------------------------------> Axisymmetry Condition
        j = 0;
        for (int ki = 1; ki<Nz-1; ki++)
        {
            i = ki-j*Nz;
            A_z_j  = M_PI*pow((r[j]+(step_r/2)),2);
            A_r_jp = 2*M_PI*(r[j]+step_r/2)*step_z;
            //--> plasma potential: phi
            mean_01_z_p = e*(ne[j][i+1]*mu_e_para[j][i+1] + ne[j][i]*mu_e_para[j][i] )/2.;
            mean_01_z_m = e*(ne[j][i-1]*mu_e_para[j][i-1] + ne[j][i]*mu_e_para[j][i] )/2.;
            mean_01_r_p = e*(ne[j+1][i]*mu_e_perp[j+1][i] + ne[j][i]*mu_e_perp[j][i] )/2.;
            C_ij  = -(1./step_z)*A_z_j*(mean_01_z_p+mean_01_z_m) - (1./step_r)*A_r_jp*mean_01_r_p;
            C_ipj =  (1./step_z)*A_z_j*mean_01_z_p;
            C_imj =  (1./step_z)*A_z_j*mean_01_z_m;
            C_ijp =  (1./step_r)*A_r_jp*mean_01_r_p;
            //-->
            entries.push_back(  T(ki, ki,    C_ij ) );
            entries.push_back(  T(ki, ki+1,  C_ipj) );
            entries.push_back(  T(ki, ki+Nz, C_ijp) );
            entries.push_back(  T(ki, ki-1,  C_imj) );

            //--> SOLUTION VECTOR
            mean_02_z_p = e*(D_e_para[j][i+1] + D_e_para[j][i])/2.;
            mean_02_z_m = e*(D_e_para[j][i-1] + D_e_para[j][i])/2.;
            mean_02_r_p = e*(D_e_perp[j+1][i] + D_e_perp[j][i])/2.;

            F_ip2j = mean_02_z_p*(1./step_z)*(ne[j][i+1] - ne[j][i]);
            F_im2j = mean_02_z_m*(1./step_z)*(ne[j][i]   - ne[j][i-1]);
            F_ijp2 = mean_02_r_p*(1./step_r)*(ne[j+1][i] - ne[j][i]);

            mean_03_z_p = e*( (D_e_para[j][i+1]*(ne[j][i+1]/Te[j][i+1])) + (D_e_para[j][i]*(ne[j][i]/Te[j][i])) )/2.;
            mean_03_z_m = e*( (D_e_para[j][i-1]*(ne[j][i-1]/Te[j][i-1])) + (D_e_para[j][i]*(ne[j][i]/Te[j][i])) )/2.;
            mean_03_r_p = e*( (D_e_perp[j+1][i]*(ne[j+1][i]/Te[j+1][i])) + (D_e_perp[j][i]*(ne[j][i]/Te[j][i])) )/2.;

            G_ip2j = mean_03_z_p*(1./step_z)*(Te[j][i+1] - Te[j][i]);
            G_im2j = mean_03_z_m*(1./step_z)*(Te[j][i]   - Te[j][i-1]);
            G_ijp2 = mean_03_r_p*(1./step_r)*(Te[j+1][i] - Te[j][i]);

            I_ip2j = e*max(Vz[j][i], 0.0)*ne[j][i]       + e*min(Vz[j][i], 0.0)*ne[j][i+1];
            I_im2j = e*max(Vz[j][i-1], 0.0)*ne[j][i-1]   + e*min(Vz[j][i-1], 0.0)*ne[j][i];
            I_ijp2 = e*max(Vr[j][i-1], 0.0)*ne[j][i]     + e*min(Vr[j][i-1], 0.0)*ne[j+1][i];
            //-->
            vect_b(ki) = (  A_z_j*(I_ip2j - I_im2j) + A_r_jp*(I_ijp2)
                        + A_z_j*(F_ip2j - F_im2j) + A_r_jp*(F_ijp2)
                        + A_z_j*(G_ip2j - G_im2j) + A_r_jp*(G_ijp2) );
                        
        }

        ///---------------------------------------------------> Sheath plan
        i = 0;
        for (int kj = 0; kj<Nre*Nz; kj+=Nz) // Electrode 
        {
            j = kj/Nz;
            if (j == 0)
            {
                A_z_j  = M_PI*pow((r[j]+(step_r/2)),2);
            }
            else
            {
                A_z_j  = M_PI*( pow((r[j]+(step_r/2)),2) - pow((r[j]-(step_r/2)),2) );
            }
            //--------------------------> Non_linear Neumann condition
            j_is = (e*ne[j][0]*sqrt(e*Te[j][0]/mi));
            Nj  = j_is*( 1 + (j_eth/j_is) ); // - exp(Lambda + ((phi_e-phi_save[kj])/Te[j][0])) );
            dNj = 0.; //(j_is/Te[j][0])*exp(Lambda + ((phi_e-phi_save[kj])/Te[j][0]));

            //--------------------------> Acutalization, Maxtrix and vector
            mean_01_z_p = e*(ne[j][1]*mu_e_para[j][1]  + ne[j][0]*mu_e_para[j][0] )/2.;
            C_ij  = -A_z_j*( (1./step_z)*mean_01_z_p + dNj);
            C_ipj = A_z_j*(1./step_z)*mean_01_z_p;
            //-->
            entries.push_back(  T(kj, kj,    C_ij ) );
            entries.push_back(  T(kj, kj+1,    C_ipj ) );

            //--------------------------> Actualisation vector
            I_ip2j = e*max(Vz[j][0], 0.0)*ne[j][0]       + e*min(Vz[j][0], 0.0)*ne[j][1];
            mean_02_z_p = e*(D_e_para[j][1] + D_e_para[j][0])/2.;
            F_ip2j = mean_02_z_p*(1./step_z)*(ne[j][1] - ne[j][0]);
            mean_03_z_p = e*( (D_e_para[j][1]*(ne[j][1]/Te[j][1])) + (D_e_para[j][0]*(ne[j][0]/Te[j][0])) )/2.;
            G_ip2j = mean_03_z_p*(1./step_z)*(Te[j][1] - Te[j][0]);
            //-->
            vect_b(kj) = A_z_j*( Nj - dNj*phi_save[kj] + I_ip2j + F_ip2j + G_ip2j);
        }

        for (int kj = Nre*Nz; kj < K-Nz; kj+=Nz) // Drop
        {
            j = kj/Nz;
            A_z_j  = M_PI*( pow((r[j]+(step_r/2)),2) - pow((r[j]-(step_r/2)),2) );
            //--------------------------> Non_linear Neumann condition
            Nj  = 0.0;
            dNj = 0.0;

            //--------------------------> Acutalization, Maxtrix and vector
            mean_01_z_p = e*(ne[j][1]*mu_e_para[j][1]  + ne[j][0]*mu_e_para[j][0] )/2.;
            C_ij  = -A_z_j*( (1./step_z)*mean_01_z_p + dNj);
            C_ipj = A_z_j*(1./step_z)*mean_01_z_p;
            //-->
            entries.push_back(  T(kj, kj,    C_ij ) );
            entries.push_back(  T(kj, kj+1,    C_ipj ) );

            //--------------------------> Actualisation vector
            I_ip2j = e*max(Vz[j][0], 0.0)*ne[j][0]       + e*min(Vz[j][0], 0.0)*ne[j][1];
            mean_02_z_p = e*(D_e_para[j][1] + D_e_para[j][0])/2.;
            F_ip2j = mean_02_z_p*(1./step_z)*(ne[j][1] - ne[j][0]);
            mean_03_z_p = e*( (D_e_para[j][1]*(ne[j][1]/Te[j][1])) + (D_e_para[j][0]*(ne[j][0]/Te[j][0])) )/2.;
            G_ip2j = mean_03_z_p*(1./step_z)*(Te[j][1] - Te[j][0]);
            //-->
            vect_b(kj) = A_z_j*( I_ip2j + F_ip2j + G_ip2j);
        }

        ///---------------------------------------------------> Mid plan
        i = Nz-1;
        for (int kj = Nz-1; kj < K-Nz; kj+=Nz)
        {
            j = (kj - i)/Nz;
            if (j == 0)
            {
                A_z_j  = M_PI*pow((r[j]+(step_r/2)),2);
                A_r_jp = 2*M_PI*(r[j]+step_r/2)*(step_z/2.);
                A_r_jm = 0.;
                //-->
                mean_01_z_m = e*(ne[j][i-1]*mu_e_para[j][i-1] + ne[j][i]*mu_e_para[j][i])/2.;
                mean_01_r_p = e*(ne[j+1][i]*mu_e_perp[j+1][i] + ne[j][i]*mu_e_perp[j][i])/2.;
                //-->
                C_ij  = -(1./step_z)*A_z_j*mean_01_z_m - (1./step_r)*A_r_jp*mean_01_r_p;
                C_imj =  (1./step_z)*A_z_j*mean_01_z_m;
                C_ijp =  (1./step_r)*A_r_jp*mean_01_r_p;
                //-->
                entries.push_back( T(kj, kj,   C_ij ) );
                entries.push_back( T(kj, kj+Nz, C_ijp) );
                entries.push_back( T(kj, kj-1, C_imj) );
            
                //--> SOLUTION VECTOR
                mean_02_z_m = e*(D_e_para[j][i-1] + D_e_para[j][i])/2.;
                mean_02_r_p = e*(D_e_perp[j+1][i] + D_e_perp[j][i])/2.;
                mean_02_r_m = 0.;
                //-->
                mean_03_z_m = e*( (D_e_para[j][i-1]*(ne[j][i-1]/Te[j][i-1])) + (D_e_para[j][i]*(ne[j][i]/Te[j][i])) )/2.;
                mean_03_r_p = e*( (D_e_perp[j+1][i]*(ne[j+1][i]/Te[j+1][i])) + (D_e_perp[j][i]*(ne[j][i]/Te[j][i])) )/2.;
                mean_03_r_m = 0.;
                //-->
                I_im2j = e*max(Vz[j][i-1], 0.0)*ne[j][i-1] + e*min(Vz[j][i-1], 0.0)*ne[j][i];
                I_ijp2 = e*max(Vr[j][i-1], 0.0)*ne[j][i]   + e*min(Vr[j][i-1], 0.0)*ne[j+1][i];
                I_ijm2 = 0.;
                //-->
                F_im2j = mean_02_z_m*(1./step_z)*(ne[j][i]   - ne[j][i-1]);
                F_ijp2 = mean_02_r_p*(1./step_r)*(ne[j+1][i] - ne[j][i]);
                F_ijm2 = 0.;
                //-->
                G_im2j = mean_03_z_m*(1./step_z)*(Te[j][i]   - Te[j][i-1]);
                G_ijp2 = mean_03_r_p*(1./step_r)*(Te[j+1][i] - Te[j][i]);
                G_ijm2 = 0.;
            }
            else
            {
                A_z_j  = M_PI*( pow((r[j]+(step_r/2)),2) - pow((r[j]-(step_r/2)),2) );
                A_r_jp = 2*M_PI*(r[j]+step_r/2)*(step_z/2.);
                A_r_jm = 2*M_PI*(r[j]-step_r/2)*(step_z/2.);
                //-->
                mean_01_z_m = e*(ne[j][i-1]*mu_e_para[j][i-1] + ne[j][i]*mu_e_para[j][i])/2.;
                mean_01_r_p = e*(ne[j+1][i]*mu_e_perp[j+1][i] + ne[j][i]*mu_e_perp[j][i])/2.;
                mean_01_r_m = e*(ne[j-1][i]*mu_e_perp[j-1][i] + ne[j][i]*mu_e_perp[j][i])/2.;
                //-->
                C_ij  = -(1./step_z)*A_z_j*mean_01_z_m - (1./step_r)*( A_r_jp*mean_01_r_p + A_r_jm*mean_01_r_m);
                C_imj =  (1./step_z)*A_z_j*mean_01_z_m;
                C_ijp =  (1./step_r)*A_r_jp*mean_01_r_p;
                C_ijm =  (1./step_r)*A_r_jm*mean_01_r_m;
                //-->
                entries.push_back( T(kj, kj,    C_ij ) );
                entries.push_back( T(kj, kj+Nz, C_ijp) );
                entries.push_back( T(kj, kj-1,  C_imj) );
                entries.push_back( T(kj, kj-Nz, C_ijm) );

                //--> SOLUTION VECTOR
                mean_02_z_m = e*(D_e_para[j][i-1] + D_e_para[j][i])/2.;
                mean_02_r_p = e*(D_e_perp[j+1][i] + D_e_perp[j][i])/2.;
                mean_02_r_m = e*(D_e_perp[j-1][i] + D_e_perp[j][i])/2.;
                //-->
                mean_03_z_m = e*( (D_e_para[j][i-1]*(ne[j][i-1]/Te[j][i-1])) + (D_e_para[j][i]*(ne[j][i]/Te[j][i])) )/2.;
                mean_03_r_p = e*( (D_e_perp[j+1][i]*(ne[j+1][i]/Te[j+1][i])) + (D_e_perp[j][i]*(ne[j][i]/Te[j][i])) )/2.;
                mean_03_r_m = e*( (D_e_perp[j-1][i]*(ne[j-1][i]/Te[j-1][i])) + (D_e_perp[j][i]*(ne[j][i]/Te[j][i])) )/2.;
                //-->
                I_im2j = e*max(Vz[j][i-1], 0.0)*ne[j][i-1]   + e*min(Vz[j][i-1], 0.0)*ne[j][i];
                I_ijp2 = e*max(Vr[j][i-1], 0.0)*ne[j][i]     + e*min(Vr[j][i-1], 0.0)*ne[j+1][i];
                I_ijm2 = e*max(Vr[j-1][i-1], 0.0)*ne[j-1][i] + e*min(Vr[j-1][i-1], 0.0)*ne[j][i];

                //-->
                F_im2j = mean_02_z_m*(1./step_z)*(ne[j][i]   - ne[j][i-1]);
                F_ijp2 = mean_02_r_p*(1./step_r)*(ne[j+1][i] - ne[j][i]);
                F_ijm2 = mean_02_r_m*(1./step_r)*(ne[j][i] - ne[j-1][i]);
                //-->
                G_im2j = mean_03_z_m*(1./step_z)*(Te[j][i]   - Te[j][i-1]);
                G_ijp2 = mean_03_r_p*(1./step_r)*(Te[j+1][i] - Te[j][i]);
                G_ijm2 = mean_03_r_m*(1./step_r)*(Te[j][i] - Te[j-1][i]);
            }
            vect_b(kj) = (- A_z_j*(I_im2j) + A_r_jp*I_ijp2 - A_r_jm*I_ijm2
                        - A_z_j*(F_im2j) + A_r_jp*F_ijp2 - A_r_jm*F_ijm2
                        - A_z_j*(G_im2j) + A_r_jp*G_ijp2 - A_r_jm*G_ijm2 );
        }

        ///---------------------------------------------------> GROUND
        j = Nr-1;
        for (int ki = K-Nz; ki<K; ki++)
        {
            i = ki - j*Nz;
            entries.push_back(  T(ki, ki, 1) );
            vect_b(ki) = 0.0;
        }

        ///---------------------------------------------------> INSIDE THE DOMAINE
        i = 1;
        j = 1;
        for (auto k = begin(vect_InsideDomain); k!=end(vect_InsideDomain); ++k)
        {
            i = int(*k) - j*Nz;
            A_z_j  = M_PI*( pow((r[j]+(step_r/2)),2) - pow((r[j]-(step_r/2)),2) );
            A_r_jp = 2*M_PI*(r[j]+step_r/2)*step_z;
            A_r_jm = 2*M_PI*(r[j]-step_r/2)*step_z;
            //--> plasma potential: phi
            mean_01_z_p = e*(ne[j][i+1]*mu_e_para[j][i+1] + ne[j][i]*mu_e_para[j][i] )/2.;
            mean_01_z_m = e*(ne[j][i-1]*mu_e_para[j][i-1] + ne[j][i]*mu_e_para[j][i] )/2.;
            mean_01_r_p = e*(ne[j+1][i]*mu_e_perp[j+1][i] + ne[j][i]*mu_e_perp[j][i] )/2.;
            mean_01_r_m = e*(ne[j-1][i]*mu_e_perp[j-1][i] + ne[j][i]*mu_e_perp[j][i] )/2.;
            C_ij  = -(1./step_z)*A_z_j*(mean_01_z_p+mean_01_z_m) - (1./step_r)*( A_r_jp*mean_01_r_p + A_r_jm*mean_01_r_m );
            C_ipj =  (1./step_z)*A_z_j*mean_01_z_p;
            C_imj =  (1./step_z)*A_z_j*mean_01_z_m;
            C_ijp =  (1./step_r)*A_r_jp*mean_01_r_p;
            C_ijm =  (1./step_r)*A_r_jm*mean_01_r_m;
            //-->
            entries.push_back( T(*k, *k,    C_ij ) );
            entries.push_back( T(*k, *k+1,  C_ipj) );
            entries.push_back( T(*k, *k+Nz, C_ijp) );
            entries.push_back( T(*k, *k-1,  C_imj) );
            entries.push_back( T(*k, *k-Nz, C_ijm) );

            //--> SOLUTION VECTOR
            //--> density
            mean_02_z_p = e*(D_e_para[j][i+1] + D_e_para[j][i])/2.;
            mean_02_z_m = e*(D_e_para[j][i-1] + D_e_para[j][i])/2.;
            mean_02_r_p = e*(D_e_perp[j+1][i] + D_e_perp[j][i])/2.;
            mean_02_r_m = e*(D_e_perp[j-1][i] + D_e_perp[j][i])/2.;
            F_ip2j = mean_02_z_p*(1./step_z)*(ne[j][i+1] - ne[j][i]);
            F_im2j = mean_02_z_m*(1./step_z)*(ne[j][i]   - ne[j][i-1]);
            F_ijp2 = mean_02_r_p*(1./step_r)*(ne[j+1][i] - ne[j][i]);
            F_ijm2 = mean_02_r_m*(1./step_r)*(ne[j][i]   - ne[j-1][i]);
            //--> density
            mean_03_z_p = e*( (D_e_para[j][i+1]*(ne[j][i+1]/Te[j][i+1])) + (D_e_para[j][i]*(ne[j][i]/Te[j][i])) )/2.;
            mean_03_z_m = e*( (D_e_para[j][i-1]*(ne[j][i-1]/Te[j][i-1])) + (D_e_para[j][i]*(ne[j][i]/Te[j][i])) )/2.;
            mean_03_r_p = e*( (D_e_perp[j+1][i]*(ne[j+1][i]/Te[j+1][i])) + (D_e_perp[j][i]*(ne[j][i]/Te[j][i])) )/2.;
            mean_03_r_m = e*( (D_e_perp[j-1][i]*(ne[j-1][i]/Te[j-1][i])) + (D_e_perp[j][i]*(ne[j][i]/Te[j][i])) )/2.;
            G_ip2j = mean_03_z_p*(1./step_z)*(Te[j][i+1] - Te[j][i]);
            G_im2j = mean_03_z_m*(1./step_z)*(Te[j][i]   - Te[j][i-1]);
            G_ijp2 = mean_03_r_p*(1./step_r)*(Te[j+1][i] - Te[j][i]);
            G_ijm2 = mean_03_r_m*(1./step_r)*(Te[j][i]   - Te[j-1][i]);
            //-->
            I_ip2j = e*max(Vz[j][i], 0.0)*ne[j][i]       + e*min(Vz[j][i], 0.0)*ne[j][i+1];
            I_im2j = e*max(Vz[j][i-1], 0.0)*ne[j][i-1]   + e*min(Vz[j][i-1], 0.0)*ne[j][i];
            I_ijp2 = e*max(Vr[j][i-1], 0.0)*ne[j][i]     + e*min(Vr[j][i-1], 0.0)*ne[j+1][i];
            I_ijm2 = e*max(Vr[j-1][i-1], 0.0)*ne[j-1][i] + e*min(Vr[j-1][i-1], 0.0)*ne[j][i];
            //-->
            vect_b(*k) = ( A_z_j*(I_ip2j - I_im2j) + A_r_jp*(I_ijp2) - A_r_jm*(I_ijm2)
                    + A_z_j*(F_ip2j - F_im2j) + A_r_jp*(F_ijp2) - A_r_jm*(F_ijm2)
                    + A_z_j*(G_ip2j - G_im2j) + A_r_jp*(G_ijp2) - A_r_jm*(G_ijm2) );

            if (int(*k+2)%Nz == 0)
            {
                j++;
            }
        }
        SparseMatrix <double> matrix_B(K,K);
        matrix_B.setFromTriplets(entries.begin(), entries.end());
        SparseLU < SparseMatrix <double> > solver_2;
        solver_2.compute(matrix_B);
        phi = solver_2.solve(vect_b);

        //--------------------------> Error
        for(int k(1); k<K-1; k++)
        {
            STD_phi_01 += abs(phi[k]  - phi_save[k]);
            STD_phi_02 += abs(phi[k]);
        }
        err_L1_phi = STD_phi_01/STD_phi_02;

        //--------------------------> Save potential
        copy_VectorXdType(phi_save, phi);
        
        ///===================================================================================================
        ///=====================: OPTION Evolution of electrode biased =======================================
        ///===================================================================================================
        if (evo_electrode_bias == 1)
        {
            if (iteration > 1000)
            {
                i = 0;
                for (int j=0; j<Nre; j++)
                {
                    k = i + j*Nz;
                    j_sh[j]   = (e*ne[j][0]*sqrt(e*Te[j][0]/mi));
                    phi_sh[j] = phi[k];
                    Te_sh[j]  = Te[j][0];
                }
                pair<double, double> result =  broot::bisect([I_k, j_sh, phi_sh, Te_sh, j_eth, Lambda, Nre](double x) { return find_electrode_bias(x, I_k, j_sh, phi_sh, Te_sh, j_eth, Lambda, Nre); }, -1000, 1000, boost::math::tools::eps_tolerance<double>());
                phi_e = (result.first + result.second) / 2.;
            }
        }
        ///===================================================================================================
        ///=====================: OUTPUT Current Density =====================================================
        ///===================================================================================================
        ///---------------------------------------------------> Sheath
        k = 0;
        for (int j=0; j<Nre; j++)
        {
            k = j*Nz;
            j_z_sh_electrode[j] = (    (e*max(Vz[j][0], 0.0)*ne[j][0]       + e*min(Vz[j][0], 0.0)*ne[j][1])
                                        -  (e*(ne[j][1]*mu_e_para[j][1]  + ne[j][0]*mu_e_para[j][0] )/2.)*(1./step_z)*(phi[k+1] - phi[k])
                                        +  (e*(D_e_para[j][1] + D_e_para[j][0])/2.)*(1./step_z)*(ne[j][1] - ne[j][0])
                                        +  (e*( (D_e_para[j][1]*(ne[j][1]/Te[j][1])) + (D_e_para[j][0]*(ne[j][0]/Te[j][0])) )/2.)*(1./step_z)*(Te[j][1] - Te[j][0])
                                   );
            I_total += j_z_sh_electrode[j]*Sr[j];
        }
		
		///===================================================================================================
		///=====================: ION VELOCITY SOLVER ========================================================
		///===================================================================================================
		i = 0;
		j = 0;
		for (int j = 0; j < Nr-1; j++)
		{
			for (int i = 0; i < Nz-1; i++)
			{
				k = i + j*(Nz-1);
				mean_nu_ei_vpara = (nu_ei[j][i+1] + nu_ei[j][i])/2.;
				mean_S_ne_vpara  = ( (S[j][i+1]/ne[j][i+1]) + (S[j][i]/ne[j][i]) )/2.;
				//-->
				mean_nu_ei_vperp = (nu_ei[j+1][i+1] + nu_ei[j][i+1])/2.;
				mean_S_ne_vperp  = ( (S[j+1][i+1]/ne[j+1][i+1]) + (S[j][i+1]/ne[j][i+1]) )/2.;
				//-->
				nu_I_perp = nu_in + mean_S_ne_vperp; //eta*mean_nu_ei_vperp + nu_in + mean_S_ne_vperp;
				nu_I_para = nu_in + mean_S_ne_vpara; //eta*mean_nu_ei_vpara + nu_in + mean_S_ne_vpara;
                if (inertial_effect == 1)
                {
                    //-------------------------------------------------------------------------------------------------> j = 0
                    if (j == 0)
                    {
                        //--> no cross term (radial)
                        Vr_dr_Vr     = (min(Vr_save[j+1][i]+Vr_save[j][i], 0.)/2.)*((Vr_save[j+1][i] - Vr_save[j][i])/(step_r))        + (max(Vr_save[j][i]+Vr_axi, 0.)/2.)*((Vr_save[j][i] - Vr_axi)/(step_r/2.));         
                        Vr_dr_Vtheta = (min(Vr_save[j+1][i]+Vr_save[j][i], 0.)/2.)*((Vtheta_save[j+1][i]- Vtheta_save[j][i])/(step_r)) + (max(Vr_save[j][i]+Vr_axi, 0.)/2.)*((Vtheta_save[j][i] - Vtheta_axi)/(step_r/2.)); 
                        
                        if (i == 0)
                        {
                            //--> mean velocity - version 01
                            Vz_mean    = ((Vz_save[j][i+1] + Vz_save[j+1][i+1] )/2.) + ((Vz_save[j][i]   + Vz_save[j+1][i] )/2.); 
                            Vr_mean    = Vr_save[j][i];
                            //--> mean velocity - version 02
                            Vz_mean_ip = (Vz_save[j][i+1] + Vz_save[j+1][i+1] )/2.; 
                            Vz_mean_im = (Vz_save[j][i]   + Vz_save[j+1][i] )/2.; 
                            Vr_mean_jp = Vr_save[j][i];
                            Vr_mean_jm = 0.;                                                                 
                            //--> cross term 
                            Vz_dz_Vr     = min(Vz_mean_ip, 0.)*((Vr_save[j][i+1] - Vr_save[j][i])/(step_z))         + max(Vz_mean_im, 0.)*((Vr_save[j][i] - Vr_sh)/(step_z));           
                            Vz_dz_Vtheta = min(Vz_mean_ip, 0.)*((Vtheta_save[j][i+1] - Vtheta_save[j][i])/(step_z)) + max(Vz_mean_im, 0.)*((Vtheta_save[j][i] - Vtheta_sh)/(step_z));   
                            Vr_dr_Vz     = 0.;         
                            //--> no cross term (axial)
                            Vz_dz_Vz     = (min(Vz_save[j][i+1]+Vz_save[j][i], 0.)/2.)*((Vz_save[j][i+1] - Vz_save[j][i])/(step_z))   + (max(Vz_save[j][i] + Vz_sh, 0.)/2.)*((Vz_save[j][i] - Vz_sh)/(step_z/2));   
                        }
                        else if (i == Nz-2)
                        {
                            //--> mean velocity - version 01
                            Vz_mean    = ((Vz_save[j][i]   + Vz_save[j+1][i] )/2.); 
                            Vr_mean    = (Vr_save[j][i] + Vr_save[j][i-1])/2.; 
                            //--> mean velocity - version 02
                            Vz_mean_ip = 0.; 
                            Vz_mean_im = (Vz_save[j][i] + Vz_save[j+1][i])/2.; 
                            Vr_mean_jp = (Vr_save[j][i] + Vr_save[j][i-1])/2.;
                            Vr_mean_jm = 0.;                                             
                            //--> cross term
                            Vz_dz_Vr     = 0.; 
                            Vz_dz_Vtheta = 0.; 
                            Vr_dr_Vz     = 0.;                                                                              
                            //--> no cross term (axial)
                            Vz_dz_Vz     = (min(Vz_mid + Vz_save[j][i], 0.)/2.)*((Vz_mid - Vz_save[j][i])/(step_z/2))          + (max(Vz_save[j][i] + Vz_save[j][i-1], 0.)/2.)*((Vz_save[j][i] - Vz_save[j][i-1])/(step_z));   
                        }
                        else if (i !=0 && i != Nz-2)
                        {
                            //--> mean velocity - version 01
                            Vz_mean    = (Vz_save[j][i]   + Vz_save[j][i+1] + Vz_save[j+1][i+1] + Vz_save[j+1][i])/4.;   
                            Vr_mean    = (Vr_save[j][i]   + Vr_save[j][i-1])/2.;  
                            //--> mean velocity - version 02
                            Vz_mean_ip = (Vz_save[j][i+1] + Vz_save[j+1][i+1] )/2.; 
                            Vz_mean_im = (Vz_save[j][i]   + Vz_save[j+1][i] )/2.;                           
                            //--> cross term
                            Vz_dz_Vr     = min(Vz_mean_ip, 0.)*((Vr_save[j][i+1] - Vr_save[j][i])/(step_z))         + max(Vz_mean_im, 0.)*((Vr_save[j][i] - Vr_save[j][i-1])/(step_z));          
                            Vz_dz_Vtheta = min(Vz_mean_ip, 0.)*((Vtheta_save[j][i+1] - Vtheta_save[j][i])/(step_z)) + max(Vz_mean_im, 0.)*((Vtheta_save[j][i] - Vtheta_save[j][i-1])/(step_z));  
                            Vr_dr_Vz     = 0.;                                                                                     
                            //--> no cross term (axial)
                            Vz_dz_Vz     = (min(Vz_save[j][i+1] + Vz_save[j][i], 0.)/2.)*((Vz_save[j][i+1] - Vz_save[j][i])/(step_z))   + (max(Vz_save[j][i] + Vz_save[j][i-1], 0.)/2.)*((Vz_save[j][i] - Vz_save[j][i-1])/(step_z));   
                        }
                    }

                    //-------------------------------------------------------------------------------------------------> j = Nr-2
                    else if (j == Nr-2)
                    { 
                        Vr_dr_Vr     = (min(Vr_ground+Vr_save[j][i], 0.)/2.)*((Vr_ground - Vr_save[j][i])/(step_r/2))        + (max(Vr_save[j][i]+Vr_save[j-1][i], 0.)/2.)*((Vr_save[j][i] - Vr_save[j-1][i])/(step_r));         
                        Vr_dr_Vtheta = (min(Vr_ground+Vr_save[j][i], 0.)/2.)*((Vtheta_ground- Vtheta_save[j][i])/(step_r/2)) + (max(Vr_save[j][i]+Vr_save[j-1][i], 0.)/2.)*((Vtheta_save[j][i] - Vtheta_save[j-1][i])/(step_r)); 
                        if (i == 0)
                        {
                            //--> mean velocity - version 01
                            Vz_mean    = (Vz_save[j][i] + Vz_save[j][i+1] )/2.;   
                            Vr_mean    = (Vr_save[j-1][i] + Vr_save[j][i])/2.;   
                            //--> mean velocity - version 02
                            Vz_mean_ip = Vz_save[j][i+1]; 
                            Vz_mean_im = Vz_save[j][i];                                   
                            Vr_mean_jp = Vr_save[j][i];     
                            Vr_mean_jm = Vr_save[j-1][i];                                          
                            //--> cross term
                            Vz_dz_Vr     = min(Vz_mean_ip, 0.)*((Vr_save[j][i+1] - Vr_save[j][i])/(step_z))         + max(Vz_mean_im, 0.)*((Vr_save[j][i] - Vr_sh)/(step_z));           
                            Vz_dz_Vtheta = min(Vz_mean_ip, 0.)*((Vtheta_save[j][i+1] - Vtheta_save[j][i])/(step_z)) + max(Vz_mean_im, 0.)*((Vtheta_save[j][i] - Vtheta_sh)/(step_z));   
                            Vr_dr_Vz     = min(Vr_mean_jp, 0.)*((Vz_ground - Vz_save[j][i])/(step_r))               + max(Vr_mean_jm, 0.)*((Vz_save[j][i] - Vz_save[j-1][i])/(step_r));  
                            //--> no cross term (axial)
                            Vz_dz_Vz     = (min(Vz_save[j][i+1]+Vz_save[j][i], 0.)/2.)*((Vz_save[j][i+1] - Vz_save[j][i])/(step_z))   + (max(Vz_save[j][i] + Vz_sh, 0.)/2.)*((Vz_save[j][i] - Vz_sh)/(step_z/2));   
                        }
                        else if (i == Nz-2)
                        {
                            //--> mean velocity - version 01
                            Vz_mean    = Vz_save[j][i];                                                                    
                            Vr_mean    = (Vr_save[j-1][i-1] + Vr_save[j-1][i] + Vr_save[j][i] +  Vr_save[j][i-1] )/4.;
                            //--> mean velocity - version 02
                            Vr_mean_jp = (Vr_save[j][i]     + Vr_save[j][i-1] )/2.;     
                            Vr_mean_jm = (Vr_save[j-1][i-1] + Vr_save[j-1][i] )/2.;     
                            //--> cross term
                            Vz_dz_Vr     = 0.;        
                            Vz_dz_Vtheta = 0.;
                            Vr_dr_Vz     = min(Vr_mean_jp, 0.)*((Vz_ground - Vz_save[j][i])/(step_r))               + max(Vr_mean_jm, 0.)*((Vz_save[j][i] - Vz_save[j-1][i])/(step_r));          
                            //--> no cross term (axial)
                            Vz_dz_Vz     = (min(Vz_mid + Vz_save[j][i], 0.)/2.)*((Vz_mid - Vz_save[j][i])/(step_z/2))          + (max(Vz_save[j][i] + Vz_save[j][i-1], 0.)/2.)*((Vz_save[j][i] - Vz_save[j][i-1])/(step_z));    
                        }
                        else if (i !=0 && i != Nz-2)
                        {
                            //--> mean velocity
                            Vz_mean = (Vz_save[j][i] + Vz_save[j][i+1] )/2.;                                            
                            Vr_mean = (Vr_save[j-1][i-1] + Vr_save[j-1][i] + Vr_save[j][i] +  Vr_save[j][i-1] )/4.;     
                            Vr_mean_jp = (Vr_save[j][i]     + Vr_save[j][i-1] )/2.;     
                            Vr_mean_jm = (Vr_save[j-1][i-1] + Vr_save[j-1][i] )/2.;       
                            //--> cross term
                            Vz_dz_Vr     = min(Vz_mean_ip, 0.)*((Vr_save[j][i+1] - Vr_save[j][i])/(step_z))            + max(Vz_mean_im, 0.)*((Vr_save[j][i] - Vr_save[j][i-1])/(step_z));         
                            Vz_dz_Vtheta = min(Vz_mean_ip, 0.)*((Vtheta_save[j][i+1] - Vtheta_save[j][i])/(step_z))    + max(Vz_mean_im, 0.)*((Vtheta_save[j][i] - Vtheta_save[j][i-1])/(step_z));  
                            Vr_dr_Vz     = min(Vr_mean_jp, 0.)*((Vz_ground - Vz_save[j][i])/(step_r))                  + max(Vr_mean_jm, 0.)*((Vz_save[j][i] - Vz_save[j-1][i])/(step_r));         
                            //--> no cross term (axial)
                            Vz_dz_Vz     = (min(Vz_save[j][i+1] + Vz_save[j][i], 0.)/2.)*((Vz_save[j][i+1] - Vz_save[j][i])/(step_z))   + (max(Vz_save[j][i] + Vz_save[j][i-1], 0.)/2.)*((Vz_save[j][i] - Vz_save[j][i-1])/(step_z));   
                        }
                    }
                    //-------------------------------------------------------------------------------------------------> j != 0 and j != Nr-2
                    else if (j != 0 && j != Nr-2)
                    {
                        Vr_dr_Vr     = (min(Vr_save[j+1][i]+Vr_save[j][i], 0.)/2.)*((Vr_save[j+1][i] - Vr_save[j][i])/(step_r))         + (max(Vr_save[j][i]+Vr_save[j-1][i], 0.)/2.)*((Vr_save[j][i] - Vr_save[j-1][i])/(step_r));         
                        Vr_dr_Vtheta = (min(Vr_save[j+1][i]+Vr_save[j][i], 0.)/2.)*((Vtheta_save[j+1][i] - Vtheta_save[j][i])/(step_r)) + (max(Vr_save[j][i]+Vr_save[j-1][i], 0.)/2.)*((Vtheta_save[j][i] - Vtheta_save[j-1][i])/(step_r)); 
                        
                        if (i == 0)
                        {
                            //--> mean velocity - version 01
                            Vz_mean    = (Vz_save[j][i] + Vz_save[j][i+1] + Vz_save[j+1][i+1] + Vz_save[j+1][i])/4.;  
                            Vr_mean    = (Vr_save[j-1][i] + Vr_save[j][i])/2.;  
                            //--> mean velocity - version 02  
                            Vz_mean_ip = (Vz_save[j][i+1] + Vz_save[j+1][i+1] )/2.; 
                            Vz_mean_im = (Vz_save[j][i]   + Vz_save[j+1][i] )/2.;      
                            Vr_mean_jp = Vr_save[j][i];     
                            Vr_mean_jm = Vr_save[j-1][i];                                         
                            //--> cross term
                            Vz_dz_Vr     = min(Vz_mean_ip, 0.)*((Vr_save[j][i+1] - Vr_save[j][i])/(step_z))         + max(Vz_mean_im, 0.)*((Vr_save[j][i] - Vr_sh)/(step_z));           
                            Vz_dz_Vtheta = min(Vz_mean_ip, 0.)*((Vtheta_save[j][i+1] - Vtheta_save[j][i])/(step_z)) + max(Vz_mean_im, 0.)*((Vtheta_save[j][i] - Vtheta_sh)/(step_z));   
                            Vr_dr_Vz     = min(Vr_mean_jp, 0.)*((Vz_save[j+1][i] - Vz_save[j][i])/(step_r))         + max(Vr_mean_jm, 0.)*((Vz_save[j][i] - Vz_save[j-1][i])/(step_r)); 
                            //--> no cross term (axial)
                            Vz_dz_Vz     = (min(Vz_save[j][i+1]+Vz_save[j][i], 0.)/2.)*((Vz_save[j][i+1] - Vz_save[j][i])/(step_z))   + (max(Vz_save[j][i] + Vz_sh, 0.)/2.)*((Vz_save[j][i] - Vz_sh)/(step_z/2));     
                        }
                        else if (i == Nz-2)
                        {
                            //--> mean velocity - version 01
                            Vz_mean    = (Vz_save[j][i] + Vz_save[j+1][i])/2.;               
                            Vr_mean    = (Vr_save[j-1][i-1] + Vr_save[j-1][i] + Vr_save[j][i] +  Vr_save[j][i-1] )/4.;     
                            //--> mean velocity - version 02                          
                            Vr_mean_jp = (Vr_save[j][i]     + Vr_save[j][i-1] )/2.;     
                            Vr_mean_jm = (Vr_save[j-1][i-1] + Vr_save[j-1][i] )/2.;  
                            //--> cross term
                            Vz_dz_Vr     = 0.;        
                            Vz_dz_Vtheta = 0.;
                            Vr_dr_Vz     = min(Vr_mean_jp, 0.)*((Vz_save[j+1][i] - Vz_save[j][i])/(step_r))         + max(Vr_mean_jm, 0.)*((Vz_save[j][i] - Vz_save[j-1][i])/(step_r));          
                            //--> no cross term (axial)
                            Vz_dz_Vz     = (min(Vz_mid + Vz_save[j][i], 0.)/2.)*((Vz_mid - Vz_save[j][i])/(step_z/2))          + (max(Vz_save[j][i] + Vz_save[j][i-1], 0.)/2.)*((Vz_save[j][i] - Vz_save[j][i-1])/(step_z));  
                        }
                        else if (i !=0 && i != Nz-2)
                        {
                            //--> mean velocity - version 01
                            Vz_mean = (Vz_save[j][i] + Vz_save[j][i+1] + Vz_save[j+1][i+1] + Vz_save[j+1][i])/4.;
                            Vr_mean    = (Vr_save[j-1][i-1] + Vr_save[j-1][i] + Vr_save[j][i] +  Vr_save[j][i-1] )/4.; 
                            //--> mean velocity - version 02    
                            Vz_mean_ip = (Vz_save[j][i+1]   + Vz_save[j+1][i+1])/2.; 
                            Vz_mean_im = (Vz_save[j][i]     + Vz_save[j+1][i]  )/2.;  
                            Vr_mean_jp = (Vr_save[j][i]     + Vr_save[j][i-1]  )/2.;     
                            Vr_mean_jm = (Vr_save[j-1][i-1] + Vr_save[j-1][i]  )/2.;  
                            //--> cross term
                            Vz_dz_Vr     = min(Vz_mean_ip, 0.)*((Vr_save[j][i+1] - Vr_save[j][i])/(step_z))         + max(Vz_mean_im, 0.)*((Vr_save[j][i] - Vr_save[j][i-1])/(step_z));         
                            Vz_dz_Vtheta = min(Vz_mean_ip, 0.)*((Vtheta_save[j][i+1] - Vtheta_save[j][i])/(step_z)) + max(Vz_mean_im, 0.)*((Vtheta_save[j][i] - Vtheta_save[j][i-1])/(step_z));  
                            Vr_dr_Vz     = min(Vr_mean_jp, 0.)*((Vz_save[j+1][i] - Vz_save[j][i])/(step_r))         + max(Vr_mean_jm, 0.)*((Vz_save[j][i] - Vz_save[j-1][i])/(step_r));         
                            //--> no cross term (axial)
                            Vz_dz_Vz     = (min(Vz_save[j][i+1] + Vz_save[j][i], 0.)/2.)*((Vz_save[j][i+1] - Vz_save[j][i])/(step_z))   + (max(Vz_save[j][i] + Vz_save[j][i-1], 0.)/2.)*((Vz_save[j][i] - Vz_save[j][i-1])/(step_z));   
                        }
                    }
                }
                //------------------------------------------------------------------------------------------------->
                //--> Radial velocity
                Vr[j][i]     = ( 1/(1 + step_t*nu_I_perp) )*( Vr_save[j][i] - inertial_effect*step_t*Vr_dr_Vr + step_t*Omega_i*Vtheta_save[j][i] + step_t*Omega_i*Vtheta_save[j][i]*inertial_effect*(Vtheta_save[j][i]/(r_v[j]*Omega_i))       
                                                                -inertial_effect*step_t*Vz_dz_Vr - ((e*step_t)/(mi*step_r))*(phi[k+1+j+Nz]-phi[k+1+j]) );
                //--> Axial velocity
                Vz[j][i]     = ( 1/(1 + step_t*nu_I_para) )*( Vz_save[j][i] - inertial_effect*step_t*Vz_dz_Vz - inertial_effect*step_t*Vr_dr_Vz - ((e*step_t)/(mi*step_z))*(phi[k+1+j] - phi[k+j])    );
                //--> Azimuthal velocity
                Vtheta[j][i] = ( 1/(1 + step_t*nu_I_perp) )*( Vtheta_save[j][i] - Vtheta_save[j][i]*inertial_effect*(step_t/r_v[j])*Vr[j][i] - step_t*Omega_i*Vr[j][i] - inertial_effect*step_t*Vr_dr_Vtheta
                                                            -inertial_effect*step_t*Vz_dz_Vtheta );  
			}
		}

		copy_MatrixType(Vr_save,     Vr);
		copy_MatrixType(Vtheta_save, Vtheta);
		copy_MatrixType(Vz_save,     Vz);
		///=======================================================================================================================================================
		///=====================: DENSITY SOLVER =================================================================================================================
		///=======================================================================================================================================================
        if (Eq_continuity == 1)
        {
            ///---------------------------------------------------> Axisymmetry Condition
            j = 0;
            for (int i = 1; i<Nz-1; i++)
            {
                A_z_j  = M_PI*pow((r[j]+(step_r/2.)),2.);
                A_r_jp = 2.*M_PI*(r[j]+step_r/2.)*step_z;
                V_ij   = A_z_j*step_z;
                //-->
                I_ip2j = max(Vz[j][i], 0.0)*ne_save[j][i]     + min(Vz[j][i], 0.0)*ne_save[j][i+1];
                I_im2j = max(Vz[j][i-1], 0.0)*ne_save[j][i-1] + min(Vz[j][i-1], 0.0)*ne_save[j][i];
                I_ijp2 = max(Vr[j][i-1], 0.0)*ne_save[j][i]     + min(Vr[j][i-1], 0.0)*ne_save[j+1][i];
                //-->
                ne[j][i] = ne_save[j][i] - (step_t/V_ij)*A_z_j*(I_ip2j-I_im2j) - (step_t/V_ij)*A_r_jp*I_ijp2 + step_t*S[j][i];
            }
                ///---------------------------------------------------> Sheath plan - ion-sheath modelling + voltage drop
            i = 0;
            for (int j = 0; j<Nr; j++)
            {
                if (j == 0)
                {
                    A_z_j  = M_PI*pow((r[j]+(step_r/2)),2.);
                    V_ij   = A_z_j*(step_z/2.);
                    //-->
                    I_ip2j = max(Vz[j][i], 0.0)*ne_save[j][i]   + min(Vz[j][i], 0.0)*ne_save[j][i+1];
                    I_im2j = (Vz_sh)*ne_save[j][i];
                    //-->
                    ne[j][i] = ne_save[j][i] - (step_t/V_ij)*A_z_j*(I_ip2j-I_im2j) + step_t*S[j][i];
                }
                else if (j == Nr-1)
                {
                    ne[j][i] = (ne_save[j-1][i] + ne_save[j][i+1])/2. ;
                }
                else
                {
                    A_z_j  = M_PI*( pow(r[j]+(step_r/2),2) - pow((r[j]-(step_r/2)),2) );
                    V_ij   = A_z_j*(step_z/2.);
                    //-->
                    I_ip2j = max(Vz[j][i], 0.0)*ne_save[j][i]   + min(Vz[j][i], 0.0)*ne_save[j][i+1];
                    I_im2j = (Vz_sh)*ne_save[j][i];
                    //-->
                    ne[j][i] = ne_save[j][i] - (step_t/V_ij)*A_z_j*(I_ip2j-I_im2j) + step_t*S[j][i];
                }

            }
                ///---------------------------------------------------> Mid plan
            i = Nz-1;
            for (int j = 0; j<Nr; j++)
            {
                if (j == 0)
                {
                    A_z_j  = M_PI*pow((r[j]+(step_r/2)),2);
                    A_r_jp = 2*M_PI*(r[j]+step_r/2)*(step_z/2.);
                    A_r_jm = 0.;
                    V_ij = A_z_j*(step_z/2.);
                    //-->
                    I_ip2j = 0.0;
                    I_im2j = max(Vz[j][i-1], 0.0)*ne_save[j][i-1] + min(Vz[j][i-1], 0.0)*ne_save[j][i];
                    I_ijp2 = max(Vr[j][i-1], 0.0)*ne_save[j][i]   + min(Vr[j][i-1], 0.0)*ne_save[j+1][i];
                    I_ijm2 = 0.0;
                }
                else if (j == Nr-1)
                {
                    A_z_j  = M_PI*( pow(r[j], 2.) - pow((r[j]-(step_r/2)),2) );
                    A_r_jp = 2*M_PI*r[j]*(step_z/2.);
                    A_r_jm = 2*M_PI*(r[j]-step_r/2)*(step_z/2.);
                    V_ij = A_z_j*(step_z/2.);
                    //-->
                    I_ip2j = 0.0;
                    I_im2j = 0.0;
                    I_ijp2 = (Vr_ground)*ne_save[j][i];
                    I_ijm2 = max(Vr[j-1][i-1], 0.0)*ne_save[j-1][i] + min(Vr[j-1][i-1], 0.0)*ne_save[j][i];
                }
                else
                {
                    A_z_j  = M_PI*( pow((r[j]+(step_r/2)),2) - pow((r[j]-(step_r/2)),2) );
                    A_r_jp = 2*M_PI*(r[j]+step_r/2)*(step_z/2.);
                    A_r_jm = 2*M_PI*(r[j]-step_r/2)*(step_z/2.);
                    V_ij = A_z_j*(step_z/2.);
                    //-->
                    I_ip2j = 0.0;
                    I_im2j = max(Vz[j][i-1], 0.0)*ne_save[j][i-1]   + min(Vz[j][i-1], 0.0)*ne_save[j][i];
                    I_ijp2 = max(Vr[j][i-1], 0.0)*ne_save[j][i]     + min(Vr[j][i-1], 0.0)*ne_save[j+1][i];
                    I_ijm2 = max(Vr[j-1][i-1], 0.0)*ne_save[j-1][i] + min(Vr[j-1][i-1], 0.0)*ne_save[j][i];
                    //-->
                }
                ne[j][i] = ne_save[j][i] - (step_t/V_ij)*A_z_j*(I_ip2j-I_im2j) - (step_t/V_ij)*A_r_jp*I_ijp2 + (step_t/V_ij)*A_r_jm*I_ijm2 + step_t*S[j][i];
            }
            ///---------------------------------------------------> GROUND
            j = Nr-1;
            for (int i = 1; i<Nz-1; i++)
            {
                A_r_jp = 2*M_PI*r[j]*step_z;
                A_r_jm = 2*M_PI*(r[j]-step_r/2)*step_z;
                V_ij = A_z_j*step_z;
                //-->
                I_ijp2 = ne_save[j][i]*Vr_ground;
                I_ijm2 = max(Vr[j-1][i-1], 0.0)*ne_save[j-1][i] + min(Vr[j-1][i-1], 0.0)*ne_save[j][i];
                //-->
                ne[j][i] = ne_save[j][i] - (step_t/V_ij)*A_r_jp*I_ijp2 + (step_t/V_ij)*A_r_jm*I_ijm2 + step_t*S[j][i];
            }
            //-----------------------------------> INSIDE THE DOMAINE <---------------------------------------------------------------------------------------
            i = 1;
            j = 1;
            for (int j = 1; j < Nr-1; j++)
            {
                for (int i = 1; i < Nz-1; i++)
                {
                    A_z_j  = M_PI*( pow((r[j]+(step_r/2)),2) - pow((r[j]-(step_r/2)),2) );
                    A_r_jp = 2*M_PI*(r[j]+step_r/2)*step_z;
                    A_r_jm = 2*M_PI*(r[j]-step_r/2)*step_z;
                    V_ij = A_z_j*step_z;
                    //-->
                    I_ip2j = max(Vz[j][i], 0.0)*ne_save[j][i]       + min(Vz[j][i], 0.0)*ne_save[j][i+1];
                    I_im2j = max(Vz[j][i-1], 0.0)*ne_save[j][i-1]   + min(Vz[j][i-1], 0.0)*ne_save[j][i];
                    I_ijp2 = max(Vr[j][i-1], 0.0)*ne_save[j][i]     + min(Vr[j][i-1], 0.0)*ne_save[j+1][i];
                    I_ijm2 = max(Vr[j-1][i-1], 0.0)*ne_save[j-1][i] + min(Vr[j-1][i-1], 0.0)*ne_save[j][i];
                    //-->
                    ne[j][i] = ne_save[j][i] - (step_t/V_ij)*A_z_j*(I_ip2j-I_im2j) - (step_t/V_ij)*A_r_jp*I_ijp2 + (step_t/V_ij)*A_r_jm*I_ijm2 + step_t*S[j][i];
                }
            }
            ///===================================================================================================
            ///=====================: Error =======================================================
            ///===================================================================================================
            for(int j(1); j<Nr-1; j++)
            {
                for(int i(1); i<Nz-1; i++)
                {
                    STD_ne_01 += abs(ne[j][i]  - ne_save[j][i]);
                    STD_ne_02 += abs(ne[j][i]);
                }
            }
            err_L1_ne = STD_ne_01/STD_ne_02;
            copy_MatrixType(ne_save, ne);
        }

		///===================================================================================================
		///=====================: Output =====================================================================
		///===================================================================================================
		if (iteration % iteration_print == 0)
		{
            cout.precision(5);
			cout << "t = " << iteration*step_t*1e3 ;
            cout.precision(dbl::max_digits10);
            if (Eq_continuity == 1)
            {
                cout << " ms, mean(ne) = " << STD_ne_02/K << ", Delta_mean(ne) = " << abs(STD_ne_02/K - mean_ne_save)  << ", err_L1_ne = " << err_L1_ne << ", err_L1_phi = " << err_L1_phi << endl;
            }
            cout << " ms, err_L1_phi = " << err_L1_phi << endl;
            if (evo_electrode_bias == 1)
            {
                cout << "phi_e = " << phi_e << " V, and I = " << I_total << " A" << endl;
            }
            // mean_ne_save = STD_ne_02/K;
		}
		if (iteration % iteration_save == 0)
		{
			Map < MatrixXd, RowMajor> phi_matrix( phi.data(), Nz,Nr);
			saveData_Macho(phi_matrix.transpose(), "phi_iteration=" + to_string(iteration)+"_Macho.csv", path_iteration, Nz, Nr);
			outputCSV_Macho(ne, "ne_iteration=" + to_string(iteration) + "_Macho.csv", path_iteration);
			outputCSV_Macho(Vr, "Vr_iteration=" + to_string(iteration) + "_Macho.csv", path_iteration);
			outputCSV_Macho(Vtheta, "Vtheta_iteration=" + to_string(iteration) + "_Macho.csv", path_iteration);
			outputCSV_Macho(Vz, "Vz_iteration=" + to_string(iteration) + "_Macho.csv", path_iteration);
            //-->
			//-->
			//outputCSV_list_Macho(j_z_sh_electrode, "j_z_sh_electrode_iteration=" + to_string(iteration) + "_cpp.csv", path_iteration);
            //outputCSV_list_Macho(j_z_sh_electrode_BC, "j_z_sh_electrode_BC_iteration=" + to_string(iteration) + "_cpp.csv", path_iteration);
            //outputCSV_list_Macho(j_z_sh_drop, "j_z_sh_drop_iteration=" + to_string(iteration) + "_cpp.csv", path_iteration);
		}
		///===================================================================================================
		///=====================: Remove memory access =======================================================
		///===================================================================================================
        if (break_time == 1)
        {
            if (iteration*step_t*1e3 > max_time_ms)
            {
                err_L1_ne = eps*1e-2;
            }
        }
		///---------------------------------->
		iteration++;	
    }
    cout << "iteration = " << iteration << ", err_L1_ne = " << err_L1_ne << ", err_L1_phi = " << err_L1_phi << endl;
	Map < MatrixXd, RowMajor> phi_matrix( phi.data(), Nz,Nr);
	saveData_Macho(phi_matrix.transpose(), "phi_CTS.csv", path_final, Nz, Nr);
	//-->
	outputCSV_Macho(ne, "ne_CTS.csv", path_final);
	outputCSV_Macho(Vr, "Vr_CTS.csv", path_final);
	outputCSV_Macho(Vtheta, "Vtheta_CTS.csv", path_final);
	outputCSV_Macho(Vz, "Vz_CTS.csv", path_final);
	//-->
    printf("END TIME: %.2fs\n", (double)(clock() - t_all)/CLOCKS_PER_SEC);
	return 0;
}
//=====================================================================================================================

bool outputCSV_Properties(double mi, double nN, double Omega_e, double Omega_i, double nu_en_0, double nu_ei_0, double nu_in, double step_t)
{

	ofstream out("Output/Test1/Properties.csv");	//open file for writing
	if (!out)
	{
		cerr<<"Could not open output file!"<<endl;
		return false;
	}
	out << "mi" << "," << mi;
	out << "\n";
	out << "nN"  << "," << nN;
	out << "\n";
	out << "Omega_e" << "," << Omega_e;
	out << "\n";
	out << "Omega_i" << "," << Omega_i;
	out << "\n";
	out << "nu_en_0" << "," << nu_en_0;
	out << "\n";
	out << "nu_ei_0" << "," << nu_ei_0;
	out << "\n";
	out << "nu_in" << "," << nu_in;

	return true;
}

bool outputCSV_list(VectorXd& list_data, string fileName,  string path)
{
	ofstream out(path + "/" + fileName);
	out.precision(22);
	if (!out)
	{
		cerr<<"Could not open output file!"<<endl;
		return false;
	}
    int rows=list_data.size();
    for(int k(0); k<rows; k++)
    {
        out<<list_data[k] << ","; //write values
    }
	return true;
}

bool outputCSV_list_Macho(VectorXd& list_data, string fileName,  string path)
{
	ofstream out(path + "/" + fileName);
	out.precision(22);
	if (!out)
	{
		cerr<<"Could not open output file!"<<endl;
		return false;
	}
    int rows=list_data.size();
    for(int k(0); k<rows; k++)
    {
        out<<list_data[k] << "    "; //write values
    }
	return true;
}

bool outputCSV(Matrix_type& matrix_2D, string fileName,  string path)
{
	ofstream out(path + "/" + fileName);
    out.precision(22);
	if (!out)
	{
		cerr<<"Could not open output file!"<<endl;
		return false;
	}
    int rows=matrix_2D.size();
    int cols=matrix_2D[0].size();
    for(int j(0); j<rows; j++)
    {
        for(int i(0); i<cols; i++)
        {
            if (i < cols-1)
            {
                out<<matrix_2D[j][i] << ","; //write values
            }
            else
            {
                out<<matrix_2D[j][i]; //write values
            }
//            cout <<  matrix_2D[j][i] << endl;
        }
        out<<"\n"; //new line, not using endl to avoid buffer flush
    }
	//file closed automatically when "out" variable is destroyed
	return true;
}

bool outputCSV_Macho(Matrix_type& matrix_2D, string fileName,  string path)
{
	ofstream out(path + "/" + fileName);
    out.precision(22);
	if (!out)
	{
		cerr<<"Could not open output file!"<<endl;
		return false;
	}
    int rows=matrix_2D.size();
    int cols=matrix_2D[0].size();
    out << cols << "    " << rows  << "\n";
    for(int j(0); j<rows; j++)
    {
        for(int i(0); i<cols; i++)
        {
            if (i < cols-1)
            {
                out<<matrix_2D[j][i] << "    "; //write values
            }
            else
            {
                out<<matrix_2D[j][i]; //write values
            }
//            cout <<  matrix_2D[j][i] << endl;
        }
        out<<"\n"; //new line, not using endl to avoid buffer flush
    }
	//file closed automatically when "out" variable is destroyed
	return true;
}

void Input_2Darray(Matrix_type& m, string fileName)
{
    ifstream myfile;
    myfile.open(fileName);
    string line, word;
    int i = 0;
    int j = 0;
    while (getline(myfile,line))
    {
        stringstream ss(line);
        vector<string> row;
        i = 0;
        while (getline(ss, word, ','))
        {
            m[j][i] = stod(word);
            i++;
        }
        j++;
    }
}


void copy_MatrixType(Matrix_type& matrix_2D_save, Matrix_type& matrix_2D)
{
    int rows=matrix_2D.size();
    int cols=matrix_2D[0].size();
    for(int j(0); j<rows; j++)
    {
        for(int i(0); i<cols; i++)
        {
            matrix_2D_save[j][i] = matrix_2D[j][i];
        }
    }
}

void copy_VectorXdType(VectorXd& phi_save, VectorXd& phi)
{
	int K_tot=phi.size();
	for(int k(0); k<K_tot; k++)
	{
		phi_save[k] = phi[k];
	}
}


bool outputCSV_scalar(double scalar, string fileName,  string path)
{
	ofstream out(path + "/" + fileName);
    if (!out)
    {
	cerr<<"Could not open output file!"<<endl;
	return false;
    }
    out<<scalar;
    return true;
}



void saveData(MatrixXd matrix, string fileName,  string path)
{
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
	ofstream file(path + "/" + fileName);
    if (file.is_open())
    {
        file << matrix.format(CSVFormat);
        file.close();
    }
}


void saveData_Macho(MatrixXd matrix, string fileName,  string path, int cols, int rows)
{
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, "    ", "\n");
	ofstream file(path + "/" + fileName);
    if (file.is_open())
    {
        file << cols << "    " << rows  << "\n";
        file << matrix.format(CSVFormat);
        file.close();
    }
}

void linspace(dvector &linear_vector, double v_start, double v_end, int node)
{
    double step = (v_end-v_start)/(node-1);
    for (int i = 0; i < node; i++ )
    {
        linear_vector[i] = v_start + i*step;
    }
}

void definition_Surface(dvector &Sr, dvector &r, double step_r, double re, double rg, int Nr, int Nre)
{
    Sr[0] = M_PI*pow((step_r/2), 2);
    for(int i(1); i<Nre-1; ++i)
    {
        Sr[i]  = 2*M_PI*r[i]*step_r;
    }
    Sr[Nre-1]    = 2*M_PI*re *(step_r/2);
    for(int i(Nre); i<Nr-1; ++i)
    {
        Sr[i]  = 2*M_PI*r[i]*step_r;
    }
    Sr[Nr-1] = 2*M_PI*rg*(step_r/2);
}

void initialization_density(Matrix_type& ne, double ne_background, dvector &ne_init, string fileName)
{
//    cout << fileName << endl;
    int Nr=ne.size();
    int Nz=ne[0].size();
    for(int j=0; j<Nr; j++)
    {
      for(int i=0; i<Nz; i++)
      {
          ne[j][i] = ne_background + ne_background*ne_init[i];
      }
    }
}

void display_VectorXd(VectorXd x, string text)
{
    cout << text << endl << x << endl;
}

void display_2Darray(Matrix_type& m, string fileName)
{
    cout << fileName << endl;
   int M=m.size();
   int N=m[0].size();
   for(int i=0; i<M; i++) {
      for(int j=0; j<N; j++)
         cout << m[i][j] << " ";
      cout << endl;
   }
   cout << endl;
}

map<string, double> loadParameters(const char *  inputFile)
{
	string tmpName, string_useless, name_variable;
	double tmpValue;
	fstream fileParameters;

	map<string, double> listOfParameters;
	fileParameters.open( inputFile , ifstream::in);
	//cout << "coucou" << endl;
    	if (!fileParameters)
    	{
      		cerr << "ERROR: The input file couldn't be opened.\n";
        	exit(1);
    	}
	int cpt = 0;
    	while ( fileParameters >> tmpName )
    	{
    		if (tmpName.at(0) != '/')
    		{
    			if (cpt%2 == 0)
    			{
    				name_variable = tmpName;
    			}
    			else if (cpt%2 == 1)
    			{
    				tmpValue = stod(tmpName);
    				listOfParameters[ name_variable ] = tmpValue;
    				//cout << name_variable << "  =  " << tmpValue << endl;
    			}
    			cpt++;
    		}
    	}
    	fileParameters.close();
    	return listOfParameters;
}






































