# effect positive
# from main import N_CLASSES
import os

N_PATH = 'C:/Users/user/Desktop/datasets/Child Skin Disease'
N_DATASET = os.path.join(N_PATH, 'Total_Dataset')


N_RES = 256 
N_BATCH = 32 


N_INFECTION = ['Abscess',
                'Cellulitis',
                'Chicken pox (varicella)',
                'Cutaneous larva migrans',
                'Eczema herpeticum',
                'Folliculitis',
                'Furuncle',
                'Green nail syndrome',
                'Herpes simplex infection',
                'Herpes zoster',
                'Impetigo',
                'Molluscum contagiosum',
                'Paronychia',
                'Staphylococcal scalded skin syndrome',
                'Tinea capitis',
                'Tinea corporis',
                'Tinea cruris',
                'Tinea faciale', 
                'Tinea manus',
                'Tinea pedis',
                'Verruca plana',
                'Viral exanthem',
                'Wart']

N_INFECTION = list((map(lambda x : x.lower().replace(' ', ''), N_INFECTION)))


N_CLASSES = 2
