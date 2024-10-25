## @ingroup Methods-Noise-Multi_Fidelity
# RCAIDE/Methods/Noise/Multi_Fidelity/harmonic_noise_plane.py
# 
# 
# Created:  Jul 2024, Niranjan Nanjappa

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
# RCAIDE
from RCAIDE.Framework.Core                                 import orientation_product, orientation_transpose  

# Python Package imports  
import numpy as np
from scipy.special import jv 
import scipy as sp

# ----------------------------------------------------------------------------------------------------------------------
# Compute Harmonic Noise 
# ----------------------------------------------------------------------------------------------------------------------
## @ingroup Methods-Noise-Frequency_Domain_Buildup-Rotor 
def harmonic_noise_plane(cpt,aeroacoustic_data,harmonics_blade,harmonics_load,conditions,propulsor_conditions,coordinates,rotor,settings,Noise):
    '''This computes the harmonic noise (i.e. thickness and loading noise) in the frequency domain 
    of a rotor at any angle of attack with load distribution along the blade span and blade chord. This is a 
    level 3 fidelity approach. All sources are computed using the helicoidal surface theory.

    Assumptions:
    1) Acoustic non-compactness of loads along blade chord.
    2) Acoustic non-compactness of loads along blade span.
    3) Acoustic compactness of loads along blade thickness.

    Source:
    1) Hanson, D. B. (1995). Sound from a propeller at angle of attack: a new theoretical viewpoint. 
    Proceedings - Royal Society of London, A, 449(1936).
    
    2) Hanson, D. B. "Noise radiation of propeller loading sources with angular inflow" AIAA 1990-3955.
    13th Aeroacoustics Conference. October 1990.
    
    3) Hanson, Donald B. "Helicoidal surface theory for harmonic noise of rotors in the far field."
    AIAA Journal 18.10 (1980): 1213-1220.

    3) Hubbard, Harvey H., ed. Aeroacoustics of flight vehicles: theory and practice. Vol. 1.
    NASA Office of Management, Scientific and Technical Information Program, 1991.


    Inputs: 
        harmonics_blade               - blade harmonics                                                            [Unitless]
        harmonics_load                - loading harmonics (modes within each blade harmonic mode)                  [Unitless]
        freestream                    - freestream data structure                                                  [m/s]
        angle_of_attack               - aircraft angle of attack                                                   [rad]
        position_vector               - position vector of aircraft                                                [m]
        velocity_vector               - velocity vector of aircraft                                                [m/s] 
        rotors                        - data structure of rotors                                                   [None]
        aeroacoustic_data             - data structure of acoustic data                                            [None]
        settings                      - accoustic settings                                                         [None] 
        res                           - results data structure                                                     [None] 

    Outputs 
        res.                                    *acoustic data is stored and passed in data structures*                                                                            
            SPL_prop_harmonic_bpf_spectrum       - harmonic noise in blade passing frequency spectrum              [dB]
            SPL_prop_harmonic_bpf_spectrum_dBA   - dBA-Weighted harmonic noise in blade passing frequency spectrum [dbA]                  
            SPL_prop_harmonic_1_3_spectrum       - harmonic noise in 1/3 octave spectrum                           [dB]
            SPL_prop_harmonic_1_3_spectrum_dBA   - dBA-Weighted harmonic noise in 1/3 octave spectrum              [dBA] 
            p_pref_harmonic                      - pressure ratio of harmonic noise                                [Unitless]
            p_pref_harmonic_dBA                  - pressure ratio of dBA-weighted harmonic noise                   [Unitless]


    Properties Used:
        N/A
    
    Code Convention - The number in front of a variable name indicates the number of dimensions of the variable.
                      For instance, m_5 is the 5 dimensional harmonic modes variable, m_3 is 3 dimensional harmonic modes variable
    '''     

    angle_of_attack         = conditions.aerodynamics.angles.alpha
    velocity_vector         = conditions.frames.inertial.velocity_vector 
    freestream              = conditions.freestream       
    num_h_b                 = len(harmonics_blade)
    num_h_l                 = len(harmonics_load)
    num_cpt                 = len(angle_of_attack) 
    num_mic                 = len(coordinates.X_hub[0,:,0,0,0]) 
    phi_0                   = np.array([rotor.phase_offset_angle])  # phase angle offset 
    num_sec                 = len(rotor.radius_distribution) 
    orientation             = np.array(rotor.orientation_euler_angles) * 1 
    body2thrust             = sp.spatial.transform.Rotation.from_rotvec(orientation).as_matrix() 
    commanded_thrust_vector = propulsor_conditions.commanded_thrust_vector_angle
    chord_coord             = aeroacoustic_data.chord_coord[0]
 
    # Lift and Drag - coefficients and distributions 
    fL      = aeroacoustic_data.disc_lift_distribution[cpt,:,:,:] 
    fD      = aeroacoustic_data.disc_lift_distribution[cpt,:,:,:]
    CL      = aeroacoustic_data.disc_lift_coefficient[cpt,:,:]
    CD      = aeroacoustic_data.disc_drag_coefficient[cpt,:,:]
                
    y_u_5   = np.tile(aeroacoustic_data.blade_upper_surface[None,None,None,None,:],(num_mic,num_sec,num_h_b,num_h_l,1))
    y_l_5   = np.tile(aeroacoustic_data.blade_lower_surface[None,None,None,None,:],(num_mic,num_sec,num_h_b,num_h_l,1))
    
    # DFT to get loading modes
    CL_k_2           = sp.fft.rfft(CL, axis=1)
    CD_k_2           = sp.fft.rfft(CD, axis=1)
    fL_k_3           = sp.fft.rfft(fL, axis=1)
    fD_k_3           = sp.fft.rfft(fD, axis=1) 
    
    # ----------------------------------------------------------------------------------
    # Rotational Noise - Loading Noise
    # ----------------------------------------------------------------------------------  
    # [microphones, radial distribution, blade harmonics, load harmonics, chord coordinate]  
    
    # freestream density and speed of sound
    rho_2          = np.tile(freestream.density[cpt,:,None],(num_mic,num_h_b))
    a_2            = np.tile(freestream.speed_of_sound[cpt,:,None],(num_mic,num_h_b))
    
    B              = rotor.number_of_blades
    
    # blade harmonics
    m_2            = np.tile(harmonics_blade[None,:],(num_mic,1))
    m_3            = np.tile(harmonics_blade[None,:,None],(num_mic,1,num_h_l))
    m_4            = np.tile(harmonics_blade[None,None,:,None],(num_mic,num_sec,1,num_h_l))
    m_5            = np.tile(harmonics_blade[None,None,:,None,None],(num_mic,num_sec,1,num_h_l,chord_coord))
                                                                                            
    # loading harmonics
    k_3            = np.tile(harmonics_load[None,None,:],(num_mic,num_h_b,1))
    k_4            = np.tile(harmonics_load[None,None,None,:],(num_mic,num_sec,num_h_b,1))
    k_5            = np.tile(harmonics_load[None,None,None,:,None],(num_mic,num_sec,num_h_b,1,chord_coord))
    
    
    # --------------------------------------------------------------------------------------------------------------------------
    # net angle of inclination of propeller axis wrt inertial axis - THIS HAS TO BE CHANGED TO ACCOUNT FOR ACTUAL ANGULAR INFLOW
    alpha_3        = np.tile((angle_of_attack[cpt,:] + np.arccos(body2thrust[0,0]))[:,None,None],(num_mic,num_h_b,num_h_l))
    alpha_4        = np.tile((angle_of_attack[cpt,:] + np.arccos(body2thrust[0,0]))[:,None,None,None],(num_mic,num_sec,num_h_b,num_h_l))
    alpha_5        = np.tile((angle_of_attack[cpt,:] + np.arccos(body2thrust[0,0]))[:,None,None,None,None],(num_mic,num_sec,num_h_b,num_h_l,chord_coord))
    # --------------------------------------------------------------------------------------------------------------------------
    
    # rotor angular speed
    omega_2        = np.tile(aeroacoustic_data.omega[cpt,:,None],(num_mic,num_h_b))   
    
    R              = rotor.radius_distribution
    
    # Non-dimensional radius distribution
    z_4            = np.tile((R/R[-1])[None,:,None,None],(num_mic,1,num_h_b,num_h_l))
    z_5            = np.tile((R/R[-1])[None,:,None,None,None],(num_mic,1,num_h_b,num_h_l,chord_coord))
    
    # Radial chord distribution
    c_4            = np.tile(rotor.chord_distribution[None,:,None,None],(num_mic,1,num_h_b,num_h_l))
    c_5            = np.tile(rotor.chord_distribution[None,:,None,None,None],(num_mic,1,num_h_b,num_h_l,chord_coord))
    
    MCA_4          = np.tile(rotor.mid_chord_alignment[None,:,None,None],(num_mic,1,num_h_b,num_h_l))
    
    # chord to diameter ratio
    R_tip          = rotor.tip_radius
    D              = 2*R[-1]
    B_D_4          = c_4/D
    B_D_5          = c_5/D
    
    # maximum thickness to chord ratio
    t_b            = rotor.thickness_to_chord
    t_b_4          = np.tile(t_b[None,:,None,None],(num_mic,1,num_h_b,num_h_l))
    
    # chordwise thickness distribution normalized wrt chord
    H_5            = (y_u_5 - y_l_5)/c_5
    
    
    # Rotorcraft speed and mach number
    V_2            = np.tile(np.linalg.norm(velocity_vector[cpt,:], axis=0) [None,None],(num_mic,num_h_b))
    M_2            = V_2/a_2
    M_4            = np.tile(M_2[:,None,:,None],(1,num_sec,1,num_h_l))
    M_5            = np.tile(M_2[:,None,:,None,None],(1,num_sec,1,num_h_l,chord_coord))
    
    # Rotor tip speed and mach number
    V_tip          = R_tip*omega_2                                                        
    M_t_2          = V_tip/a_2
    M_t_4          = np.tile(M_t_2[:,None,:,None],(1,num_sec,1,num_h_l))
    M_t_5          = np.tile(M_t_2[:,None,:,None,None],(1,num_sec,1,num_h_l,chord_coord))
    
    # Section relative mach number
    M_r_4          = np.sqrt(M_4**2 + (z_4**2)*(M_t_4**2))
    
    # retarded theta
    theta_r        = coordinates.theta_hub_r[cpt,:,0,0]
    theta_r_2      = np.tile(theta_r[:,None],(1,num_h_b))
    theta_r_3      = np.tile(theta_r[:,None,None],(1,num_h_b,num_h_l))
    theta_r_4      = np.tile(theta_r[:,None,None,None],(1,num_sec,num_h_b,num_h_l))
    theta_r_5      = np.tile(theta_r[:,None,None,None,None],(1,num_sec,num_h_b,num_h_l,chord_coord))
    
    # ---------------------------------------------------------------------------------------------
    # retarded distance to source - HAVE TO CHECK DEFINITION
    Y              = np.sqrt(coordinates.X_hub[cpt,:,0,0,1]**2 +  coordinates.X_hub[cpt,:,0,0,2] **2)
    Y_2            = np.tile(Y[:,None],(1,num_h_b))
    r_2            = Y_2/np.sin(theta_r_2)
    # ---------------------------------------------------------------------------------------------
    
    # phase angles
    phi_0_vec      = np.tile(phi_0[:,None,None],(num_mic,num_h_b,num_h_l))
    phi_3          = np.tile(coordinates.phi_hub_r[cpt,:,0,0,None,None],(1,num_h_b,num_h_l)) + phi_0_vec
    phi_4          = np.tile(phi_3[:,None,:,:],(1,num_sec,1,1))
    phi_5          = np.tile(phi_3[:,None,:,:,None],(1,num_sec,1,1,chord_coord))
    
    # total angle between propeller axis and r vector
    theta_r_prime_3 = np.arccos(np.cos(theta_r_3)*np.cos(alpha_3) + np.sin(theta_r_3)*np.sin(phi_3)*np.sin(alpha_3))
    theta_r_prime_4 = np.arccos(np.cos(theta_r_4)*np.cos(alpha_4) + np.sin(theta_r_4)*np.sin(phi_4)*np.sin(alpha_4))
    theta_r_prime_5 = np.arccos(np.cos(theta_r_5)*np.cos(alpha_5) + np.sin(theta_r_5)*np.sin(phi_5)*np.sin(alpha_5))
        
    phi_prime_3    = np.arccos((np.sin(theta_r_3)*np.cos(phi_3))/np.sin(theta_r_prime_3))
    
    # Velocity in the rotor frame
    T_body2inertial = conditions.frames.body.transform_to_inertial
    T_inertial2body = orientation_transpose(T_body2inertial)
    V_body          = orientation_product(T_inertial2body,velocity_vector)
    body2thrust,_   = rotor.body_to_prop_vel(commanded_thrust_vector)
    T_body2thrust   = orientation_transpose(body2thrust)
    V_thrust        = orientation_product(T_body2thrust,V_body)
    V_thrust_perp   = V_thrust[cpt,0]
    V_thrust_perp_2 = np.tile(V_thrust_perp[None,None],(num_mic,num_h_b))
    M_thrust_2      = V_thrust_perp_2/a_2
    M_thrust_4      = np.tile(M_thrust_2[:,None,:,None],(1,num_sec,1,num_h_l))
    
    # helicoid angle
    zeta_4          = np.arctan(M_thrust_4/(z_4*M_t_4))
    zeta_5          = np.tile(zeta_4[:,:,:,:,None],(1,1,1,1,chord_coord))
    
    # wavenumbers
    k_m_2          = m_2*B*omega_2/a_2
    k_m_bar        = k_m_2/(1 - M_2*np.cos(theta_r_2))
    k_x_hat_4      = 2*B_D_4*(((m_4*B-k_4)*np.cos(zeta_4))/z_4 + (m_4*B*M_t_4*np.cos(theta_r_prime_4)*np.sin(zeta_4))/(1-M_4*np.cos(theta_r_4)))
    k_x_hat_5      = 2*B_D_5*(((m_5*B-k_5)*np.cos(zeta_5))/z_5 + (m_5*B*M_t_5*np.cos(theta_r_prime_5)*np.sin(zeta_5))/(1-M_5*np.cos(theta_r_5)))
    k_y_hat_4      = 2*B_D_4*(((m_4*B-k_4)*np.sin(zeta_4))/z_4 - (m_4*B*M_t_4*np.cos(theta_r_prime_4)*np.cos(zeta_4))/(1-M_4*np.cos(theta_r_4)))
    
    # phase angles
    phi_s_4        = k_x_hat_4*MCA_4/c_4
    # NEED TO DEFINE phi_FA too (phase angle due to face alignment of the rotor blade)
    
    Noise.f[cpt,:,:]  = B*m_2*omega_2/(2*np.pi)

    
    CL_k_4         = np.tile(CL_k_2[None,:,None,0:num_h_l],(num_mic,1,num_h_b,1))
    CD_k_4         = np.tile(CD_k_2[None,:,None,0:num_h_l],(num_mic,1,num_h_b,1))
    
    # [control point, microphones, rotors, radial distribution, blade harmonics, load harmonics, chordwise coordinate]
    fL_k_5         = np.tile(fL_k_3[None,:,None,0:num_h_l,:],(num_mic,1,num_h_b,1,1))
    fD_k_5         = np.tile(fD_k_3[None,:,None,0:num_h_l,:],(num_mic,1,num_h_b,1,1))
    
    
    # frequency domain source function for drag and lift
    X_edge         = np.linspace(-0.5,0.5,chord_coord+1)
    dX             = np.diff(X_edge)
    dX_tiled_5     = np.tile(dX[None,None,None,None,:],(num_mic,num_sec,num_h_b,num_h_l,1))
    X              = 0.5*(X_edge[0:-1] + X_edge[1:])
    X_5            = np.tile(X[None,None,None,None,:],(num_mic,num_sec,num_h_b,num_h_l,1))
    exp_term_5     = np.exp(1j*k_x_hat_5*X_5)
    psi_Lk_4       = np.trapz(fL_k_5*exp_term_5, x=X, axis=4)
    psi_Dk_4       = np.trapz(fD_k_5*exp_term_5, x=X, axis=4)
    
    psi_hat_Lk_4   = psi_Lk_4*np.exp(1j*(phi_s_4 + phi_4))
    psi_hat_Dk_4   = psi_Dk_4*np.exp(1j*(phi_s_4 + phi_4))
    psi_hat_Fk_4   = 0.5*(k_y_hat_4*CL_k_4*psi_hat_Lk_4 + k_x_hat_4*CD_k_4*psi_hat_Dk_4)
    
    
    # FREQUENCY DOMAIN PRESSURE TERM FOR LOADING
    J_mBk_4        = jv(m_4*B-k_4, (m_4*B*z_4*M_t_4*np.sin(theta_r_prime_4))/(1-M_4*np.cos(theta_r_4)))
    L_Integrand_4  = (M_r_4**2)*psi_hat_Fk_4*J_mBk_4
    L_Summand_3    = np.trapz(L_Integrand_4, x=z_4[0,:,0,0], axis=1)*np.exp(1j*(m_3*B-k_3)*(phi_prime_3-(np.pi/2)))
    L_Summation_2  = np.sum(L_Summand_3, axis=2)
    P_Lm           = (-1j*rho_2*(a_2**2)*B*np.exp(1j*k_m_2*r_2)*L_Summation_2)/(4*np.pi*(r_2/R_tip)*(1-M_2*np.cos(theta_r_2)))
    
    # frequency domain source function for drag and lift
    psi_V_4        = np.trapz(H_5*exp_term_5, x=X, axis=4)
    
    # FREQUENCY DOMAIN PRESSURE TERM FOR THICKNESS
    V_Integrand_4  = (M_r_4**2)*(k_x_hat_4**2)*t_b_4*psi_V_4*J_mBk_4
    V_Summand_3    = np.trapz(V_Integrand_4, x=z_4[0,:,0,0], axis=1)*np.exp(1j*m_3*B*(phi_prime_3-(np.pi/2)))
    
    # we take a single dimension along the 4th axis because we only want the loading mode corresponding to k=0
    V_Summation_2  = V_Summand_3[:,:,0]
    P_Vm           = (-rho_2*(a_2**2)*B*np.exp(1j*k_m_2*r_2)*V_Summation_2)/(4*np.pi*(r_2/R_tip)*(1-M_2*np.cos(theta_r_2)))
    
    
    # SOUND PRESSURE LEVELS
    Noise.P_Lm_abs[cpt,:,:]     = np.abs(P_Lm)
    Noise.P_Vm_abs[cpt,:,:]     = np.abs(P_Vm)
    # Noise.SPL_prop_harmonic_bpf_spectrum     = 20*np.log10((abs(P_Lm_abs + P_Vm_abs))/p_ref)  
    # Noise.SPL_prop_harmonic_1_3_spectrum     = convert_to_third_octave_band(Noise.SPL_prop_harmonic_bpf_spectrum,Noise.f,settings)          
    # Noise.SPL_prop_harmonic_1_3_spectrum[np.isinf(Noise.SPL_prop_harmonic_1_3_spectrum)]         = 0 
    
    return
    