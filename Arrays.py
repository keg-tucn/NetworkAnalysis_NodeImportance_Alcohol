ROIs = ['Primary motor cortex R', 'Secondary motor cortex R', 'Primary somatosensory cortex R',
        'Secondary somatosensory cortex R',  'Insular cortex R', 'Agranular insular cortex R', 'Infralimbic cortex R',
        'Prelimbic cortex R', 'Piriform cortex R', 'Primary auditory cortex R', 'Secondary auditory cortex R',
        'Primary visual cortex R', 'Secondary visual cortex R', 'Cingulate cortex R', 'Orbital cortex R',
        'Retrosplenial cortex R', 'Parietal association cortex R', 'Endopiriform nucleus R', 'Subiculum R',
        'Hippocampus anterodorsal R', 'Hippocampus posterior, dorsal part R', 'Hippocampus ventral R',
        'Accumbens R', 'Caudate putamen R', 'Ventral pallidum R', 'Lateral globus pallidus R', 'Septum R',
        'Bed nucleus of the stria terminalis R', 'Ventral tegmental area R', 'Substantia nigra R', 'Raphe R',
        'Medial geniculate R', 'Mesencephalic region R', 'Superior colliculus R', 'Periaqueductal grey R',
        'Amygdala, central R', 'Amygdala, medial R', 'Thalamus, ventromedial R', 'Thalamus, dorsolateral R',
        'Thalamus, midline dorsal R', 'Hypothalamus, paraventricular R', 'Hypothalamus, lateral R', 'Primary motor cortex L',
        'Secondary motor cortex L', 'Primary somatosensory cortex L', 'Secondary somatosensory cortex L', 'Insular cortex L',
        'Agranular insular cortex L', 'Infralimbic cortex L', 'Prelimbic cortex L', 'Piriform cortex L',
        'Primary auditory cortex L', 'Secondary auditory cortex L', 'Primary visual cortex L', 'Secondary visual cortex L',
        'Cingulate cortex L', 'Orbital cortex L', 'Retrosplenial cortex L', 'Parietal association cortex L',
        'Temporal association cortex L', 'Ectorhinal cortex L', 'Endopiriform nucleus L', 'Subiculum L',
        'Hippocampus anterodorsal L', 'Hippocampus posterior, dorsal part L', 'Hippocampus ventral L', 'Accumbens L',
        'Caudate putamen L', 'Ventral pallidum L', 'Lateral globus pallidus L', 'Septum L',
        'Bed nucleus of the stria terminalis L', 'Ventral tegmental area L', 'Substantia nigra L', 'Medial geniculate L',
        'Mesencephalic region L', 'Superior colliculus L', 'Periaqueductal grey L', 'Amygdala, central L',
        'Amygdala, medial L', 'Thalamus, ventromedial L', 'Thalamus, dorsolateral L', 'Thalamus, midline dorsal L',
        'Hypothalamus, paraventricular L', 'Hypothalamus, lateral L']

orderd_control_isolates = [79, 36, 35, 30, 29, 78, 60, 40, 73, 28, 59, 72, 83, 24, 68, 6, 31, 74, 80, 7, 77, 48, 37, 49,
                           34, 50, 61, 14, 8, 21, 56, 84, 65, 71, 17, 27, 41, 25, 69, 11, 53, 9, 75, 18, 51, 5, 32, 47,
                           22, 62, 10, 4, 33, 1, 15, 46, 66, 43, 45, 76, 52, 57, 3, 26, 39, 70, 12, 13, 55, 82, 16, 67,
                           0, 2, 19, 20, 23, 38, 42, 44, 54, 58, 63, 64, 81]

orderd_stddev_control_etoh = [79, 78, 36, 30, 35, 60, 59, 68, 48, 77, 24, 47, 29, 71, 11,  6, 56,
                              5, 61, 49, 62, 27, 65, 21, 80,  7, 50, 51, 14, 17, 10,  9,  8, 52,
                              25, 37, 73, 40, 75, 70,  4, 76, 18, 46,  3, 74, 28, 53, 33, 34, 22,
                              83, 31, 82, 69, 45, 64, 26, 41, 23, 32, 81, 20, 66, 72, 67, 38, 54,
                              63, 55, 39, 84, 58, 42, 19,  2, 44, 12,  0, 16,  1, 13, 43, 15, 57]

stddev_values_control_etoh = [0.03254952, 0.0344274, 0.0322211, 0.02785887, 0.02723232, 0.02442528, 0.02427219,
                              0.02550832, 0.02649905, 0.02607125, 0.02573863, 0.0242498, 0.03248132, 0.03470652,
                              0.02563856, 0.04214779, 0.03359046, 0.02569509, 0.02756951, 0.032046, 0.02981393,
                              0.02502902, 0.02818119, 0.02936939, 0.02297346, 0.02669748, 0.02921303, 0.02488254,
                              0.02793397, 0.02366177, 0.02032534, 0.02825623, 0.02957609, 0.02795777, 0.02809334,
                              0.02156188, 0.02013601, 0.0267778, 0.03063701, 0.03152759, 0.02705046, 0.02933172,
                              0.03185075, 0.03704891, 0.03224849, 0.02874771, 0.02764889, 0.0233367, 0.02263838,
                              0.02447473, 0.0255281, 0.02553576, 0.02654735, 0.02794561, 0.03079869, 0.03105253,
                              0.02437154, 0.04319586, 0.03179073, 0.02193953, 0.02172969, 0.02443725, 0.02456897,
                              0.03085111, 0.02882198, 0.02491602, 0.03000467, 0.03025982, 0.02221871, 0.02869121,
                              0.02722345, 0.0239563, 0.03018357, 0.02689514, 0.02786272, 0.02715379, 0.02739092,
                              0.02296269, 0.01999879, 0.01891947, 0.02542734, 0.02959441, 0.02828301, 0.02825298,
                              0.03165608]

orderd_isolates_difference_etoh_control = [62, 76, 66, 65, 9, 77, 22, 47, 51, 32, 52, 8, 17, 33, 70, 39, 82, 21, 34, 37,
                                           5, 25, 74, 50, 69, 78, 54, 58, 81, 13, 16, 26, 31, 46, 55, 68, 71, 80, 3, 29,
                                           35, 48, 67, 75, 10, 12, 28, 53, 57, 73, 1, 24, 36, 41, 43, 45, 49, 61, 72,
                                           15, 4, 18, 59, 11, 14, 30, 56, 60, 84, 27, 40, 79, 83, 6, 7]

group1 = [80, 82, 76, 71, 75, 84, 81, 67, 69, 70, 64, 61]

group2 = [66, 68, 74, 63, 65, 62, 77]

group3 = [7, 29, 35, 36, 30, 40, 60, 59, 28, 73]

orderd_boxplot_div_diff_irq = [33, 62, 23, 71, 72, 73, 28, 77, 75, 41, 18, 46, 70, 4, 76, 5, 49, 34, 31, 26, 22, 65, 27,
                               74, 47, 25, 59, 67, 51, 48, 11, 84, 3, 63, 80, 32, 66, 21, 61, 56, 20, 82, 52, 17, 7, 10,
                               69, 29, 64, 2, 50, 37, 0, 81, 13, 14, 78, 42, 55, 44, 24, 68, 38, 19, 9, 8, 35, 6, 39,
                               60, 45, 36, 30, 83, 40, 58, 16, 79, 53, 12, 54, 1, 43, 15, 57]
