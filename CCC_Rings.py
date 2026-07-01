import numpy as np
import healpy as hp
import sys

# Fixed geometric configurations
NSIDE = 128
ACHATAMENTO_ALVO = 0.95
RAIO_GRAUS = 4
SIGMA_CORTE = 3.0 

PATH_MAPA = "/content/drive/MyDrive/CCC_Planck/Planck_143GHz.fits"

print("="*65)
print("PLANCK SURVEY: SEARCH FOR ANISOTROPIC CONCENTRIC STRUCTURES")
print(f"Target Geometry: Radius = {RAIO_GRAUS}°, Ellipticity = {ACHATAMENTO_ALVO}")
print("="*65)

try:
    mapa = hp.read_map(PATH_MAPA, field=0, verbose=False)
    mapa = hp.pixelfunc.ud_grade(mapa, NSIDE)
except:
    print("[ERROR] Failed to load the fits map. Check the file path.")
    sys.exit()

var_background = np.var(mapa)
NPIXELS = hp.nside2npix(NSIDE)

raio_a = np.radians(RAIO_GRAUS)
raio_b = raio_a * ACHATAMENTO_ALVO
raio_maximo_busca = raio_a * 1.1 * 1.5

# Neighborhood suppression tracking
pixels_mascarados = np.zeros(NPIXELS, dtype=bool)
estruturas_reais = 0

print(f"Analyzing {NPIXELS} sky pixels with neighborhood suppression...")
print("-" * 65)

for pixel_atual in range(NPIXELS):
    if pixels_mascarados[pixel_atual]:
        continue
        
    vizinhos = hp.query_disc(NSIDE, hp.pix2vec(NSIDE, pixel_atual), raio_maximo_busca)
    if len(vizinhos) < 50:
        continue
        
    theta_0, phi_0 = hp.pix2ang(NSIDE, pixel_atual)
    theta, phi = hp.pix2ang(NSIDE, vizinhos)
    
    d_theta = theta - theta_0
    d_phi = phi - phi_0
    
    sigmas_angulos = []
    for angulo_alpha in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        variancias_aneis = []
        for mult in [0.9, 1.0, 1.1]:
            r_a = raio_a * mult
            r_b = raio_b * mult
            
            termo1 = ((d_theta * np.cos(angulo_alpha) + d_phi * np.sin(theta_0) * np.sin(angulo_alpha))**2) / (r_a**2)
            termo2 = ((-d_theta * np.sin(angulo_alpha) + d_phi * np.sin(theta_0) * np.cos(angulo_alpha))**2) / (r_b**2)
            distancia_elipse = termo1 + termo2
            
            borda_bool = (distancia_elipse >= 0.95) & (distancia_elipse <= 1.05)
            pixels_borda = vizinhos[borda_bool]
            
            if len(pixels_borda) > 10:
                variancias_aneis.append(np.var(mapa[pixels_borda]))
                
        if len(variancias_aneis) == 3:
            sigmas_angulos.append(np.mean(variancias_aneis) / var_background)
            
    sigma_maximo = max(sigmas_angulos) if sigmas_angulos else 0
    
    if sigma_maximo >= SIGMA_CORTE:
        estruturas_reais += 1
        
        # Coordinate conversion to standard Galactic format
        lat = 90.0 - np.degrees(theta_0)
        lon = np.degrees(phi_0)
        if lon > 180: lon -= 360.0
        
        print(f"[DETECTION] Structure #{estruturas_reais}")
        print(f"   -> Central Pixel: {pixel_atual}")
        print(f"   -> Significance: {sigma_maximo:.4f} Sigma")
        print(f"   -> Coordinates: Galactic Lat = {lat:.2f}°, Galactic Lon = {lon:.2f}°")
        print("-" * 50)
        
        # Neighborhood suppression mask execution
        raio_supressao = np.radians(RAIO_GRAUS)
        pixels_para_bloquear = hp.query_disc(NSIDE, hp.pix2vec(NSIDE, pixel_atual), raio_supressao)
        pixels_mascarados[pixels_para_bloquear] = True

print("="*65)
print(f"SURVEY COMPLETE. Total independent structures resolved: {estruturas_reais}")
print("="*65)