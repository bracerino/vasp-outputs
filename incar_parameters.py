"""
VASP INCAR parameter definitions and explanations
Based on VASP documentation at https://vasp.at/wiki/
"""

INCAR_PARAMETERS = {
    'SYSTEM': {
        'description': 'Descriptive string for the system',
        'value_meaning': lambda v: f'System title: "{v}"',
        'category': 'General'
    },
    'ISTART': {
        'description': 'Job control: how to start the calculation',
        'value_meaning': lambda v: {
            '0': 'Start from scratch (read WAVECAR if present but start new SCF)',
            '1': 'Continue job - read WAVECAR file',
            '2': 'Same orbitals as in WAVECAR file',
            '3': 'Perform non-selfconsistent calculation only'
        }.get(v, f'Start mode: {v}'),
        'category': 'General'
    },
    'ICHARG': {
        'description': 'Determines how VASP constructs the initial charge density',
        'value_meaning': lambda v: {
            '0': 'Calculate charge density from initial wave functions (may cause convergence issues)',
            '1': 'Read CHGCAR and extrapolate to new positions (recommended for repeated calculations with small changes)',
            '2': 'Take superposition of atomic charge densities (default)',
            '4': 'Read potential from POT file (for OEP methods, VASP 5.1+)',
            '5': 'External charge-density-update mode (for DFT+DMFT with TRIQS)',
            '10': 'Non-selfconsistent: constant charge density throughout calculation',
            '11': 'Non-selfconsistent: fixed charge density from CHGCAR',
            '12': 'Non-selfconsistent: superposition of atomic densities (Harris-Foulkes functional)'
        }.get(v, f'Charge initialization mode: {v}'),
        'category': 'Electronic Structure'
    },
    'PREC': {
        'description': 'Precision mode for calculations',
        'value_meaning': lambda v: {
            'Low': 'Low precision - smallest FFT grid',
            'Medium': 'Medium precision - moderate FFT grid',
            'Normal': 'Normal precision (same as Medium)',
            'High': 'High precision - larger FFT grid',
            'Accurate': 'Accurate - large FFT grid, tight convergence',
            'Single': 'Single precision arithmetic'
        }.get(v, f'Precision: {v}'),
        'category': 'Precision'
    },
    'ENCUT': {
        'description': 'Energy cutoff for plane wave basis set (eV)',
        'value_meaning': lambda v: f'Plane wave cutoff: {v} eV',
        'category': 'Precision'
    },
    'EDIFF': {
        'description': 'Global break condition for electronic SC loop (eV)',
        'value_meaning': lambda v: f'SCF converges when total energy change < {v} eV',
        'category': 'Electronic Convergence'
    },
    'EDIFFG': {
        'description': 'Break condition for ionic relaxation loop',
        'value_meaning': lambda v: f'Forces < |{v}| eV/Å (negative) or energy change < {v} eV (positive)',
        'category': 'Ionic Convergence'
    },
    'ISMEAR': {
        'description': 'Determines partial occupancies for each orbital',
        'value_meaning': lambda v: {
                                       '-5': 'Tetrahedron method with Blöchl corrections (accurate DOS)',
                                       '-4': 'Tetrahedron method without Blöchl corrections',
                                       '-1': 'Fermi smearing',
                                       '0': 'Gaussian smearing',
                                       '1': 'Methfessel-Paxton order 1',
                                       '2': 'Methfessel-Paxton order 2'
                                   }.get(v,
                                         f'Smearing: {v}') + ' (use -5 for DOS, >0 for metals, 0 for semiconductors)',
        'category': 'Electronic Structure'
    },
    'SIGMA': {
        'description': 'Width of smearing (eV)',
        'value_meaning': lambda v: f'Smearing width: {v} eV',
        'category': 'Electronic Structure'
    },
    'IBRION': {
        'description': 'Determines how ions are updated and moved',
        'value_meaning': lambda v: {
            '-1': 'No update - single point calculation',
            '0': 'Molecular dynamics',
            '1': 'Quasi-Newton (RMM-DIIS) algorithm',
            '2': 'Conjugate gradient algorithm',
            '3': 'Damped molecular dynamics',
            '5': 'Finite differences - force constants',
            '6': 'Finite differences with symmetry',
            '7': 'Perturbation theory - phonons',
            '8': 'Perturbation theory with symmetry'
        }.get(v, f'Ionic update method: {v}'),
        'category': 'Ionic Relaxation'
    },
    'ISIF': {
    'description': 'Determines which degrees of freedom are allowed to change',
    'value_meaning': lambda v: {
        '0': 'Calculate forces only (no stress tensor)',
        '1': 'Calculate forces and trace of stress tensor only (no relaxation)',
        '2': 'Calculate forces and full stress tensor (default, no relaxation)',
        '3': 'Relax ions, cell shape, and cell volume (full relaxation)',
        '4': 'Relax ions and cell shape (constant volume)',
        '5': 'Change cell shape (constant volume, external stress)',
        '6': 'Change cell volume only (external stress)',
        '7': 'Calculate forces and full stress tensor (like ISIF=2, different algorithm)'
    }.get(v, f'Relaxation type: {v}'),
    'category': 'Ionic Relaxation'
    },
    'NSW': {
        'description': 'Maximum number of ionic steps',
        'value_meaning': lambda v: f'Maximum {v} ionic steps',
        'category': 'Ionic Relaxation'
    },
    'NELM': {
        'description': 'Maximum number of electronic SC steps',
        'value_meaning': lambda v: f'Maximum {v} electronic SCF steps',
        'category': 'Electronic Convergence'
    },
    'NELMIN': {
        'description': 'Minimum number of electronic SC steps',
        'value_meaning': lambda v: f'Minimum {v} electronic SCF steps',
        'category': 'Electronic Convergence'
    },
    'ALGO': {
        'description': 'Electronic minimization algorithm',
        'value_meaning': lambda v: {
            'Normal': 'Blocked Davidson iteration scheme',
            'VeryFast': 'RMM-DIIS (for large systems)',
            'Fast': 'Mix of Davidson and RMM-DIIS',
            'Conjugate': 'Conjugate gradient algorithm',
            'All': 'Simultaneous optimization of all bands',
            'Damped': 'Damped velocity friction algorithm',
            'Exact': 'Exact diagonalization'
        }.get(v, f'Algorithm: {v}'),
        'category': 'Electronic Minimization'
    },
    'LREAL': {
        'description': 'Projection in real or reciprocal space',
        'value_meaning': lambda v: {
            '.FALSE.': 'Reciprocal space (more accurate)',
            'False': 'Reciprocal space',
            '.TRUE.': 'Real space (faster for >20 atoms)',
            'True': 'Real space',
            'Auto': 'Real space, fully automatic optimization',
            'A': 'Real space, automatically optimized'
        }.get(v, f'{v}'),
        'category': 'Performance'
    },
    'LWAVE': {
        'description': 'Write WAVECAR file',
        'value_meaning': lambda v: 'WAVECAR written' if v.upper() in ['.TRUE.', 'T', 'TRUE'] else 'WAVECAR not written',
        'category': 'Output Control'
    },
    'LCHARG': {
        'description': 'Write CHGCAR and CHG files',
        'value_meaning': lambda v: 'Charge files written' if v.upper() in ['.TRUE.', 'T',
                                                                           'TRUE'] else 'Charge files not written',
        'category': 'Output Control'
    },
    'LORBIT': {
        'description': 'Determines whether DOSCAR/PROCAR files are written',
        'value_meaning': lambda v: {
            '0': 'No DOSCAR written',
            '1': 'DOSCAR with lm-decomposed DOS',
            '2': 'DOSCAR with lm-decomposed DOS + phase',
            '5': 'DOSCAR with lm-decomposed DOS (no PROCAR)',
            '10': 'DOSCAR with l-decomposed DOS',
            '11': 'DOSCAR + PROCAR with phase',
            '12': 'DOSCAR + PROCAR without phase'
        }.get(v, f'Output: {v}'),
        'category': 'Output Control'
    },
    'ISPIN': {
        'description': 'Spin polarization switch',
        'value_meaning': lambda v: 'Non-spin-polarized' if v == '1' else 'Spin-polarized (magnetism allowed)',
        'category': 'Magnetism'
    },
    'MAGMOM': {
        'description': 'Initial magnetic moment for each atom',
        'value_meaning': lambda v: f'Initial moments: {v} μB',
        'category': 'Magnetism'
    },
    'NCORE': {
        'description': 'Number of cores working on one orbital',
        'value_meaning': lambda v: f'{v} cores per band - affects parallelization',
        'category': 'Performance'
    },
    'NPAR': {
        'description': 'Number of bands treated in parallel',
        'value_meaning': lambda v: f'{v} groups of bands in parallel (deprecated, use NCORE)',
        'category': 'Performance'
    },
    'KPAR': {
        'description': 'Number of k-points treated in parallel',
        'value_meaning': lambda v: f'{v} k-point groups in parallel',
        'category': 'Performance'
    },
    'LHFCALC': {
        'description': 'Switch to turn on Hartree-Fock routines',
        'value_meaning': lambda v: 'Hybrid functional enabled' if v.upper() in ['.TRUE.', 'T',
                                                                                'TRUE'] else 'No hybrid functional',
        'category': 'Exchange-Correlation'
    },
    'GGA': {
        'description': 'Type of generalized gradient approximation',
        'value_meaning': lambda v: {
            'PE': 'PBE (Perdew-Burke-Ernzerhof)',
            'PS': 'PBEsol',
            'RP': 'revPBE (revised PBE)',
            '91': 'Perdew-Wang 91',
            'AM': 'AM05'
        }.get(v, f'GGA: {v}'),
        'category': 'Exchange-Correlation'
    },
    'IVDW': {
        'description': 'Van der Waals correction method',
        'value_meaning': lambda v: {
            '0': 'No vdW correction',
            '1': 'DFT-D2 (Grimme)',
            '10': 'DFT-D3 zero damping (Grimme)',
            '11': 'DFT-D3 BJ damping (Grimme)',
            '12': 'DFT-D3 zero damping (old)',
            '2': 'TS (Tkatchenko-Scheffler)',
            '4': 'dDsC dispersion',
            '20': 'DFTD4'
        }.get(v, f'vdW: {v}'),
        'category': 'Exchange-Correlation'
    },
    'LASPH': {
        'description': 'Include aspherical contributions from gradient corrections',
        'value_meaning': lambda v: 'Aspherical terms included (recommended for accurate energies)' if v.upper() in [
            '.TRUE.', 'T', 'TRUE'] else 'Spherical approximation',
        'category': 'Precision'
    },
    'POTIM': {
        'description': 'Time step for ionic motion or scaling factor',
        'value_meaning': lambda v: f'Scaling: {v} (for IBRION=1,2,3) or timestep (for IBRION=0)',
        'category': 'Ionic Relaxation'
    },
    'ADDGRID': {
        'description': 'Additional support grid for augmentation charges',
        'value_meaning': lambda v: 'Additional grid used (more accurate)' if v.upper() in ['.TRUE.', 'T',
                                                                                           'TRUE'] else 'Standard grid',
        'category': 'Precision'
    }
}


def get_parameter_explanation(param_name, param_value):
    if param_name in INCAR_PARAMETERS:
        param_info = INCAR_PARAMETERS[param_name].copy()
        # If value_meaning is a lambda/function, call it with the value
        if callable(param_info['value_meaning']):
            param_info['value_meaning'] = param_info['value_meaning'](param_value)
        return param_info
    else:
        return {
            'description': 'VASP parameter (description not in database)',
            'value_meaning': f'Value: {param_value}',
            'category': 'Other'
        }
