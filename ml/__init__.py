"""
Machine Learning module for Star Predictive Analysis.
Contains data preprocessing, model training, evaluation, and inference pipelines.
"""

STAR_CLASSES = [
    'Brown Dwarf',
    'Red Dwarf',
    'White Dwarf',
    'Main Sequence',
    'Supergiant',
    'Hypergiant'
]

CLASS_DESCRIPTIONS = {
    'Brown Dwarf': {
        'classification': 'Substellar object (Spectral class M/L/T)',
        'core_fusion': 'Incapable of sustaining hydrogen fusion; slowly cools over billions of years.',
        'luminosity_range': 'Very low (< 0.001 L☉)',
        'typical_radius': '0.05 - 0.15 R☉ (roughly Jupiter-sized)',
        'typical_temp': '< 3,500 K',
        'color': '#8B4513',
        'summary': 'Brown dwarfs are substellar objects that never accumulated enough mass to initiate core hydrogen fusion. Often called "failed stars", they bridge the gap between giant gas planets and low-mass red dwarfs.'
    },
    'Red Dwarf': {
        'classification': 'Low-mass main-sequence star (Spectral class M)',
        'core_fusion': 'Fuses hydrogen into helium via the p-p chain extremely slowly.',
        'luminosity_range': '0.0001 - 0.1 L☉',
        'typical_radius': '0.1 - 0.7 R☉',
        'typical_temp': '2,400 - 3,700 K',
        'color': '#ef4444',
        'summary': 'Red dwarfs are the most abundant stars in the Milky Way. Due to their low mass and slow rate of hydrogen fusion, they are fully convective and have lifespans reaching trillions of years.'
    },
    'White Dwarf': {
        'classification': 'Stellar remnant (Degenerate matter)',
        'core_fusion': 'No nuclear fusion occurring; radiates residual thermal energy.',
        'luminosity_range': '0.0001 - 0.1 L☉',
        'typical_radius': '0.008 - 0.02 R☉ (roughly Earth-sized)',
        'typical_temp': '8,000 - 40,000 K',
        'color': '#38bdf8',
        'summary': 'White dwarfs are ultra-dense stellar remnants left behind after low-to-medium mass stars exhaust their nuclear fuel and shed outer layers. Electron degeneracy pressure prevents further gravitational collapse.'
    },
    'Main Sequence': {
        'classification': 'Luminosity Class V (Dwarf/Core Hydrogen-Burning)',
        'core_fusion': 'Stable core hydrogen fusion in hydrostatic equilibrium.',
        'luminosity_range': '0.1 - 10,000 L☉',
        'typical_radius': '0.8 - 10 R☉',
        'typical_temp': '4,000 - 30,000 K',
        'color': '#f59e0b',
        'summary': 'Main sequence stars represent the longest and most stable phase of stellar evolution, where gravitational collapse is counteracted by outward thermal radiation pressure from core hydrogen fusion.'
    },
    'Supergiant': {
        'classification': 'Luminosity Class Ib / Iab',
        'core_fusion': 'Evolved high-mass star fusing heavier elements (helium, carbon, oxygen).',
        'luminosity_range': '10,000 - 200,000 L☉',
        'typical_radius': '30 - 500 R☉',
        'typical_temp': '3,000 - 25,000 K',
        'color': '#ec4899',
        'summary': 'Supergiants are massive evolved stars nearing the end of their stellar lives. With immense luminosities and expanded envelopes, they will ultimately end in core-collapse Type II supernovae.'
    },
    'Hypergiant': {
        'classification': 'Luminosity Class 0 / Ia+',
        'core_fusion': 'Extreme core fusion near the Eddington luminosity limit; severe mass loss.',
        'luminosity_range': '100,000 - 1,000,000+ L☉',
        'typical_radius': '500 - 2,000+ R☉',
        'typical_temp': '3,000 - 35,000 K',
        'color': '#a855f7',
        'summary': 'Hypergiants are the most massive, luminous, and rarest stars in the universe. Experiencing intense stellar winds and extreme mass loss, they push against the upper physical limits of stellar stability.'
    }
}
