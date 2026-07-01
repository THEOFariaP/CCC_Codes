import os, numpy as np, healpy as hp
from google.colab import drive
drive.mount('/content/drive')

# ========== CONFIG ==========
NSIDE = 256 # Commander vem em 256
TARGET_PIXEL = 111627 # pixel CCC
ECCENTRICITY = 0.95 # b/a
RADIUS_DEGREES = 4.0 # raio maior
FWHM_SUAVIZAR = 1.0 # graus - beam comum
PATH_DRIVE = "/content/drive/MyDrive/CCC_Planck/"
DUST_MAP = "COM_CompMap_Dust-commander_0256_R2.00.fits" # baixa do PLA se não tiver
# ============================

# 1. carrega poeira térmica Commander - já separada do CMB
dust_path = os.path.join(PATH_DRIVE, DUST_MAP)
dust = hp.read_map(dust_path, field=0) # MJy/sr em 353 GHz
dust = hp.ma(dust)
dust[dust == hp.UNSEEN] = np.nan
if hp.get_nside(dust)!= NSIDE:
    dust = hp.ud_grade(hp.ma(dust), NSIDE).filled(np.nan)

# 2. suaviza para beam comum
dust_smooth = hp.smoothing(dust, fwhm=np.radians(FWHM_SUAVIZAR))

# 3. geometria elíptica igual ao seu código original
theta_0, phi_0 = hp.pix2ang(NSIDE, TARGET_PIXEL)
raio_a = np.radians(RADIUS_DEGREES)
raio_b = raio_a * ECCENTRICITY
search_radius = raio_a * 1.6

vec_center = hp.ang2vec(theta_0, phi_0)
neighbors = hp.query_disc(NSIDE, vec_center, search_radius)
theta_n, phi_n = hp.pix2ang(NSIDE, neighbors)
d_theta = theta_n - theta_0
d_phi = phi_n - phi_0
d_phi = (d_phi + np.pi) % (2*np.pi) - np.pi # wrap

# 4. testa as 4 orientações, mas reporta todas - sem cherry-pick
resultados = []
for alpha in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
    t1 = ((d_theta*np.cos(alpha) + d_phi*np.sin(theta_0)*np.sin(alpha))**2) / raio_a**2
    t2 = ((-d_theta*np.sin(alpha) + d_phi*np.sin(theta_0)*np.cos(alpha))**2) / raio_b**2
    ell = t1 + t2

    edge = neighbors[(ell >= 0.9) & (ell <= 1.1)]
    bkg = neighbors[(ell >= 1.3) & (ell <= 1.6)]
    if len(edge) < 10 or len(bkg) < 10:
        continue

    mean_edge = np.nanmean(dust_smooth[edge])
    mean_bkg = np.nanmean(dust_smooth[bkg])
    std_bkg = np.nanstd(dust_smooth[bkg])

    sigma = (mean_edge - mean_bkg) / std_bkg if std_bkg > 0 else np.nan
    resultados.append(sigma)
    print(f"angulo {np.degrees(alpha):.0f}°: {sigma:.2f} sigma")

# 5. veredicto conservador
sigma_mediano = np.nanmedian(resultados)
sigma_max = np.nanmax(resultados)
print("\n" + "="*50)
print(f"Mediana entre ângulos: {sigma_mediano:.2f} sigma")
print(f"Máximo entre ângulos: {sigma_max:.2f} sigma")
print("-"*50)

if sigma_mediano < 2 and sigma_max < 3:
    print("VEREDICTO: sem excesso significativo de poeira térmica no anel.")
elif sigma_mediano > 3:
    print("VEREDICTO: excesso de poeira detectado em todas as orientações.")
else:
    print("VEREDICTO: marginal. Pode haver filamento parcial. Checar mapa visual.")