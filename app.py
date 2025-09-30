import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re

from incar_parameters import get_parameter_explanation

st.set_page_config(
    page_title="VASP Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)


hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.sidebar.title("VASP Output File Analysis")

css = '''
<style>
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.15rem !important;
    color: #1e3a8a !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 20px !important;
}

.stTabs [data-baseweb="tab-list"] button {
    background-color: #f0f4ff !important;
    border-radius: 12px !important;
    padding: 8px 16px !important;
    transition: all 0.3s ease !important;
    border: none !important;
    color: #1e3a8a !important;
}

.stTabs [data-baseweb="tab-list"] button:hover {
    background-color: #dbe5ff !important;
    cursor: pointer;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: #e0e7ff !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(30, 58, 138, 0.3) !important;
}

.stTabs [data-baseweb="tab-list"] button:focus {
    outline: none !important;
}
</style>


'''

st.markdown(css, unsafe_allow_html=True)

def parse_vasprun_dos(vasprun_content):
    try:
        from pymatgen.io.vasp.outputs import Vasprun
        from pymatgen.electronic_structure.core import Spin
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(vasprun_content)
            temp_path = f.name

        vasprun = Vasprun(temp_path, parse_dos=True, parse_eigen=False)
        complete_dos = vasprun.complete_dos
        efermi = vasprun.efermi

        os.unlink(temp_path)

        energies = complete_dos.energies - efermi

        if Spin.down in complete_dos.densities:
            spin_polarized = True
            dos_up = complete_dos.densities[Spin.up]
            dos_down = complete_dos.densities[Spin.down]
            return energies, dos_up, dos_down, efermi, spin_polarized, complete_dos
        else:
            spin_polarized = False
            dos = complete_dos.densities[Spin.up]
            return energies, dos, None, efermi, spin_polarized, complete_dos

    except Exception as e:
        raise ValueError(f"Error parsing vasprun.xml: {str(e)}")


def parse_vasprun_bands(vasprun_content, kpoints_content=None):
    try:
        from pymatgen.io.vasp.outputs import Vasprun
        from pymatgen.io.vasp.inputs import Kpoints
        from pymatgen.electronic_structure.core import Spin
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(vasprun_content)
            vasprun_path = f.name

        kpoints_path = None
        if kpoints_content:
            with tempfile.NamedTemporaryFile(mode='w', suffix='', delete=False) as f:
                f.write(kpoints_content)
                kpoints_path = f.name

        vasprun = Vasprun(vasprun_path, parse_dos=False, parse_eigen=True)

        efermi = vasprun.efermi
        if efermi is None:
            efermi = 0.0

        try:
            if kpoints_path:
                band_structure = vasprun.get_band_structure(
                    kpoints_filename=kpoints_path,
                    line_mode=True,
                    efermi=efermi
                )
            else:
                band_structure = vasprun.get_band_structure(
                    line_mode=True,
                    efermi=efermi
                )
        except:
            try:
                band_structure = vasprun.get_band_structure(
                    line_mode=False,
                    efermi=efermi
                )
            except Exception as e:
                raise ValueError(f"Could not parse band structure. Error: {str(e)}")

        os.unlink(vasprun_path)
        if kpoints_path:
            os.unlink(kpoints_path)

        return band_structure, efermi

    except Exception as e:
        raise ValueError(f"Error parsing band structure from vasprun.xml: {str(e)}")


def parse_vasprun_ionic_steps(vasprun_content):
    try:
        from pymatgen.io.vasp.outputs import Vasprun
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(vasprun_content)
            temp_path = f.name

        vasprun = Vasprun(temp_path, parse_dos=False, parse_eigen=False)

        ionic_steps_data = {
            'energies': [],
            'forces': [],
            'stresses': [],
            'structures': []
        }

        for ionic_step in vasprun.ionic_steps:
            ionic_steps_data['energies'].append(ionic_step['e_0_energy'])

            if 'forces' in ionic_step:
                forces = ionic_step['forces']
                max_force = np.max(np.linalg.norm(forces, axis=1))
                ionic_steps_data['forces'].append(max_force)

            if 'stress' in ionic_step:
                stress = ionic_step['stress']
                ionic_steps_data['stresses'].append(stress)

            if 'structure' in ionic_step:
                ionic_steps_data['structures'].append(ionic_step['structure'])

        os.unlink(temp_path)

        return ionic_steps_data, vasprun.final_structure, vasprun.initial_structure

    except Exception as e:
        raise ValueError(f"Error parsing ionic steps from vasprun.xml: {str(e)}")

def parse_oszicar(oszicar_content):
    lines = oszicar_content.strip().split('\n')

    optimization_steps = []
    ionic_steps = []
    energies = []
    electronic_steps = []
    ncg_steps = []
    de_values = []
    methods_used = []
    elec_steps_per_ionic = []

    current_ionic_step = 0
    current_ionic_elec_count = 0

    for line in lines:
        line = line.strip()

        electronic_match = re.match(
            r'^\s*(DAV|RMM):\s*(\d+)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(\d+)',
            line
        )
        if electronic_match:
            method = electronic_match.group(1)
            elec_step = int(electronic_match.group(2))
            energy = float(electronic_match.group(3))
            de = float(electronic_match.group(4)) if electronic_match.group(4) else 0
            d_eps = float(electronic_match.group(5)) if electronic_match.group(5) else 0
            ncg = int(electronic_match.group(6)) if electronic_match.group(6) else 0

            optimization_steps.append(len(optimization_steps) + 1)
            electronic_steps.append(
                f"Ionic {current_ionic_step if current_ionic_step > 0 else 1}, {method} {elec_step}")
            methods_used.append(method)
            de_values.append(abs(de))
            ncg_steps.append(ncg)

            current_ionic_elec_count += 1
            continue

        ionic_match = re.match(r'^\s*(\d+)\s+F=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
        if ionic_match:
            if current_ionic_elec_count > 0:
                elec_steps_per_ionic.append(current_ionic_elec_count)

            current_ionic_step = int(ionic_match.group(1))
            energy = float(ionic_match.group(2))
            ionic_steps.append(current_ionic_step)
            energies.append(energy)
            current_ionic_elec_count = 0
            continue

    return (optimization_steps, electronic_steps, ionic_steps, energies,
            ncg_steps, de_values, methods_used, elec_steps_per_ionic)


def parse_incar_detailed(incar_content):
    params = {}
    lines = incar_content.strip().split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line or '=' not in line:
            continue

        if '!' in line:
            line = line.split('!')[0].strip()

        parts = line.split('=')
        if len(parts) >= 2:
            key = parts[0].strip()
            value = parts[1].strip()
            if value:
                params[key] = value

    return params




def parse_incar(incar_content):
    ediff = 1E-4

    lines = incar_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        ediff_match = re.search(r'EDIFF\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line, re.IGNORECASE)
        if ediff_match:
            ediff = float(ediff_match.group(1))

    return ediff


def parse_doscar(doscar_content):
    lines = doscar_content.strip().split('\n')

    try:
        header_parts = lines[0].split()
        natoms = int(header_parts[0])

        header_line = lines[5].split()
        emax = float(header_line[0])
        emin = float(header_line[1])
        nedos = int(header_line[2])
        efermi = float(header_line[3])

        dos_data = []
        start_line = 6

        for i in range(start_line, start_line + nedos):
            if i < len(lines):
                values = lines[i].split()
                dos_data.append([float(v) for v in values])

        df = pd.DataFrame(dos_data)

        if df.shape[1] == 5:
            df.columns = ['Energy', 'DOS_up', 'DOS_down', 'Int_DOS_up', 'Int_DOS_down']
            spin_polarized = True
        elif df.shape[1] == 3:
            df.columns = ['Energy', 'DOS', 'Integrated_DOS']
            spin_polarized = False
        else:
            raise ValueError(f"Unexpected number of columns in DOSCAR: {df.shape[1]}")

        return df, efermi, spin_polarized, natoms

    except Exception as e:
        raise ValueError(f"Error parsing DOSCAR: {str(e)}")


def parse_eigenval(eigenval_content):
    lines = eigenval_content.strip().split('\n')

    try:
        nelect = int(lines[5].split()[0])
        nkpts = int(lines[5].split()[1])
        nbands = int(lines[5].split()[2])

        kpoints = []
        eigenvalues = []

        line_idx = 7

        for ik in range(nkpts):
            if line_idx >= len(lines):
                break

            kpt_line = lines[line_idx].split()
            kpt = [float(kpt_line[0]), float(kpt_line[1]), float(kpt_line[2])]
            kpoints.append(kpt)

            line_idx += 1

            bands_for_this_kpt = []
            for ib in range(nbands):
                if line_idx >= len(lines):
                    break
                band_line = lines[line_idx].split()
                if len(band_line) >= 2:
                    energy = float(band_line[1])
                    bands_for_this_kpt.append(energy)
                line_idx += 1

            eigenvalues.append(bands_for_this_kpt)
            line_idx += 1

        kpoints = np.array(kpoints)
        eigenvalues = np.array(eigenvalues)

        k_distances = np.zeros(len(kpoints))
        for i in range(1, len(kpoints)):
            k_distances[i] = k_distances[i - 1] + np.linalg.norm(kpoints[i] - kpoints[i - 1])

        return kpoints, eigenvalues, k_distances, nbands, nkpts

    except Exception as e:
        raise ValueError(f"Error parsing EIGENVAL: {str(e)}")


def identify_file_type(filename):
    filename_lower = filename.lower()
    if 'incar' in filename_lower:
        return 'INCAR'
    elif 'oszicar' in filename_lower:
        return 'OSZICAR'
    elif 'doscar' in filename_lower:
        return 'DOSCAR'
    elif 'eigenval' in filename_lower:
        return 'EIGENVAL'
    elif 'vasprun' in filename_lower:
        return 'VASPRUN'
    elif 'kpoints' in filename_lower:
        return 'KPOINTS'
    elif 'poscar' in filename_lower or 'contcar' in filename_lower:
        return 'STRUCTURE'
    else:
        return 'UNKNOWN'


st.info(
    "Upload your **VASP output files** into **sidebar** (OSZICAR, INCAR, DOSCAR, EIGENVAL, KPOINTS, POSCAR, CONTCAR, vasprun.xml) for analysis. Files are automatically recognized by name.")

uploaded_files = st.sidebar.file_uploader(
    "Upload VASP files",
    accept_multiple_files=True,
    help="Upload OSZICAR, INCAR, DOSCAR, POSCAR, CONTCAR, vasprun.xml, and/or EIGENVAL files"
)

st.sidebar.info(
        "🌀 Developed by **[IMPLANT team](https://implant.fs.cvut.cz/)**. **[Tutorial here](https://youtu.be/4-96LZoc_AU)**. Spot a bug or have a feature idea? Let us know at: "
        "**lebedmi2@cvut.cz**. To compile the app locally, visit our **[GitHub page](https://github.com/bracerino/vasp-outputs)**. If you like the app, please cite **[article in IUCr](https://journals.iucr.org/j/issues/2025/05/00/hat5006/index.html).** "
    )

if uploaded_files:
    incar_content = None
    oszicar_content = None
    doscar_content = None
    eigenval_content = None
    incar_found = False
    oszicar_found = False
    doscar_found = False
    eigenval_found = False

    #
    vasprun_content = None
    vasprun_found = False

    kpoints_content = None
    kpoints_found = False

    poscar_content = None
    contcar_content = None
    poscar_found = False
    contcar_found = False

    detected_files = []

    for uploaded_file in uploaded_files:
        file_type = identify_file_type(uploaded_file.name)

        if file_type == 'INCAR':
            incar_content = uploaded_file.getvalue().decode("utf-8")
            incar_found = True
            detected_files.append(f"INCAR ({uploaded_file.name})")
        elif file_type == 'OSZICAR':
            oszicar_content = uploaded_file.getvalue().decode("utf-8")
            oszicar_found = True
            detected_files.append(f"OSZICAR ({uploaded_file.name})")
        elif file_type == 'DOSCAR':
            doscar_content = uploaded_file.getvalue().decode("utf-8")
            doscar_found = True
            detected_files.append(f"DOSCAR ({uploaded_file.name})")
        elif file_type == 'EIGENVAL':
            eigenval_content = uploaded_file.getvalue().decode("utf-8")
            eigenval_found = True
            detected_files.append(f"EIGENVAL ({uploaded_file.name})")
        elif file_type == 'VASPRUN':
            vasprun_content = uploaded_file.getvalue().decode("utf-8")
            vasprun_found = True
            detected_files.append(f"VASPRUN ({uploaded_file.name})")

        elif file_type == 'STRUCTURE':
            if 'poscar' in uploaded_file.name.lower():
                poscar_content = uploaded_file.getvalue().decode("utf-8")
                poscar_found = True
                detected_files.append(f"POSCAR ({uploaded_file.name})")
            elif 'contcar' in uploaded_file.name.lower():
                contcar_content = uploaded_file.getvalue().decode("utf-8")
                contcar_found = True
                detected_files.append(f"CONTCAR ({uploaded_file.name})")

        elif file_type == 'KPOINTS':
            kpoints_content = uploaded_file.getvalue().decode("utf-8")
            kpoints_found = True
            detected_files.append(f"KPOINTS ({uploaded_file.name})")


    if detected_files:
        st.success("**Files successfully loaded:**\n\n" + " • ".join(detected_files))


    tabs_list = []
    if oszicar_found:
        tabs_list.append("📊 OSZICAR Analysis")
    if doscar_found:
        tabs_list.append("📈 DOSCAR Analysis")
    if vasprun_found:
        tabs_list.append("🔬 VASPRUN DOS")
        tabs_list.append("🎵 VASPRUN Bands")
        tabs_list.append("⚛️ VASPRUN Ionic Steps")
    if eigenval_found:
        tabs_list.append("🌊 Band Structure")
    if poscar_found or contcar_found:
        tabs_list.append("🔄 Structure Viewer")
    if incar_found:
        tabs_list.append("⚙️ INCAR Parameters")

    if not tabs_list:
        st.error("No valid VASP files detected. Please upload OSZICAR, DOSCAR, EIGENVAL, and/or INCAR files.")
    else:
        tabs = st.tabs(tabs_list)
        tab_idx = 0

        if oszicar_found:
            with tabs[tab_idx]:
                st.header("VASP Optimization Convergence Analysis")

                try:
                    if incar_found and incar_content:
                        ediff = parse_incar(incar_content)
                        ediff_source = "from INCAR file"
                    else:
                        ediff = 1E-4
                        ediff_source = "default value (no INCAR file provided)"

                    (optimization_steps, electronic_steps, ionic_steps, energies,
                     ncg_steps, de_values, methods_used, elec_steps_per_ionic) = parse_oszicar(oszicar_content)

                    if not ionic_steps or not energies:
                        st.error("No valid energy data found in OSZICAR file. Please check the file format.")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.info(f"**Ionic Steps Found:** {len(ionic_steps)}")
                        with col2:
                            st.info(f"**EDIFF:** {ediff:.2E} eV")
                        with col3:
                            st.info(f"**EDIFF Source:** {ediff_source}")

                        energy_diffs = []
                        if len(energies) > 1:
                            for i in range(1, len(energies)):
                                energy_diffs.append(abs(energies[i] - energies[i - 1]))
                            energy_diffs.insert(0, float('inf'))
                        else:
                            energy_diffs = [float('inf')]

                        df = pd.DataFrame({
                            'Ionic Step': ionic_steps,
                            'Energy (eV)': energies,
                            'Energy Difference (eV)': energy_diffs,
                            'Electronic Steps': elec_steps_per_ionic
                        })

                        st.subheader("Energy Convergence Analysis")

                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=ionic_steps,
                            y=energies,
                            name='Total Energy',
                            mode='lines+markers',
                            line=dict(width=3, color='blue'),
                            marker=dict(size=8),
                            yaxis='y1',
                            hovertemplate='Step: %{x}<br>Energy: %{y:.6f} eV<extra></extra>'
                        ))

                        if len(energy_diffs) > 1:
                            valid_diffs = [diff for diff in energy_diffs[1:] if diff != float('inf')]
                            if valid_diffs:
                                fig.add_trace(go.Scatter(
                                    x=ionic_steps[1:],
                                    y=valid_diffs,
                                    name='|ΔE|',
                                    mode='lines+markers',
                                    line=dict(width=2, color='red', dash='dash'),
                                    marker=dict(size=6, symbol='square'),
                                    yaxis='y2',
                                    hovertemplate='Step: %{x}<br>|ΔE|: %{y:.2E} eV<extra></extra>'
                                ))

                        if len(energy_diffs) > 1:
                            fig.add_hline(
                                y=ediff,
                                line_dash="dot",
                                line_color="green",
                                annotation_text=f"EDIFF = {ediff:.2E} eV",
                                annotation_position="top right",
                                annotation_font_size=18,
                                yref="y2"
                            )

                        fig.update_layout(
                            title=dict(
                                text="VASP Optimization Convergence",
                                font=dict(size=24, color='black')
                            ),
                            xaxis=dict(
                                title='Ionic Step',
                                title_font=dict(size=20, color='black'),
                                tickfont=dict(size=20, color='black'),
                                gridcolor='lightgray'
                            ),
                            yaxis=dict(
                                title='Total Energy (eV)',
                                side='left',
                                title_font=dict(size=20, color='blue'),
                                tickfont=dict(size=20, color='blue')
                            ),
                            yaxis2=dict(
                                title='|Energy Difference| (eV)',
                                side='right',
                                overlaying='y',
                                title_font=dict(size=20, color='red'),
                                tickfont=dict(size=20, color='red'),
                                tickformat='.1E'
                            ),
                            legend=dict(
                                x=0.5,
                                y=-0.2,
                                xanchor='center',
                                orientation='h',
                                bgcolor='rgba(255,255,255,0.8)',
                                borderwidth=1,
                                font=dict(size=18)
                            ),
                            height=600,
                            plot_bgcolor='white',
                            font=dict(size=20, color='black')
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("Electronic Convergence Efficiency")

                        col_eff1, col_eff2 = st.columns(2)

                        with col_eff1:
                            if elec_steps_per_ionic:
                                fig_elec = go.Figure()

                                fig_elec.add_trace(go.Bar(
                                    x=ionic_steps,
                                    y=elec_steps_per_ionic,
                                    name='Electronic Steps',
                                    marker_color='purple',
                                    hovertemplate='Ionic Step: %{x}<br>Electronic Steps: %{y}<extra></extra>'
                                ))

                                fig_elec.update_layout(
                                    title=dict(
                                        text="Electronic Steps per Ionic Step",
                                        font=dict(size=18, color='black')
                                    ),
                                    xaxis=dict(
                                        title='Ionic Step',
                                        title_font=dict(size=14, color='black'),
                                        tickfont=dict(size=12, color='black')
                                    ),
                                    yaxis=dict(
                                        title='Electronic Steps',
                                        title_font=dict(size=14, color='purple'),
                                        tickfont=dict(size=12, color='purple')
                                    ),
                                    height=400,
                                    plot_bgcolor='white'
                                )

                                st.plotly_chart(fig_elec, use_container_width=True)

                        with col_eff2:
                            if methods_used:
                                method_counts = pd.Series(methods_used).value_counts()

                                fig_methods = go.Figure(data=[go.Pie(
                                    labels=method_counts.index,
                                    values=method_counts.values,
                                    hole=0.3,
                                    hovertemplate='Method: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                                )])

                                fig_methods.update_layout(
                                    title=dict(
                                        text="Electronic Method Usage",
                                        font=dict(size=18, color='black')
                                    ),
                                    height=400,
                                    font=dict(size=12)
                                )

                                st.plotly_chart(fig_methods, use_container_width=True)

                        st.subheader("Convergence Analysis Summary")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                label="Total Ionic Steps",
                                value=len(ionic_steps)
                            )

                        with col2:
                            final_energy = energies[-1] if energies else 0
                            st.metric(
                                label="Final Energy (eV)",
                                value=f"{final_energy:.6f}"
                            )

                        with col3:
                            if len(energy_diffs) > 1:
                                final_diff = energy_diffs[-1] if energy_diffs[-1] != float('inf') else 0
                                st.metric(
                                    label="Final |ΔE| (eV)",
                                    value=f"{final_diff:.2E}" if final_diff != float('inf') else "N/A"
                                )
                            else:
                                st.metric(
                                    label="Final |ΔE| (eV)",
                                    value="N/A"
                                )

                        with col4:
                            if len(energy_diffs) > 1:
                                final_diff = energy_diffs[-1] if energy_diffs[-1] != float('inf') else 0
                                energy_converged = final_diff < ediff if final_diff != float('inf') else False
                                st.metric(
                                    label="Energy Converged",
                                    value="Yes" if energy_converged else "No"
                                )
                            else:
                                st.metric(
                                    label="Energy Converged",
                                    value="Unknown"
                                )

                        if len(energy_diffs) > 1:
                            valid_diffs = [diff for diff in energy_diffs[1:] if diff != float('inf')]
                            if valid_diffs:
                                converged_steps = [i + 2 for i, diff in enumerate(valid_diffs) if diff < ediff]
                                if converged_steps:
                                    st.success(
                                        f"Energy convergence first achieved at ionic step: **{converged_steps[0]}**")
                                else:
                                    st.warning(f"Energy difference has not reached EDIFF threshold of {ediff:.2E} eV")

                        if elec_steps_per_ionic:
                            avg_elec_steps = np.mean(elec_steps_per_ionic)
                            max_elec_steps = max(elec_steps_per_ionic)
                            min_elec_steps = min(elec_steps_per_ionic)

                            st.info(f"**Electronic Convergence Efficiency:** Avg: {avg_elec_steps:.1f} steps/ionic, "
                                    f"Range: {min_elec_steps}-{max_elec_steps} steps")

                        if methods_used:
                            method_counts = pd.Series(methods_used).value_counts()
                            method_summary = ", ".join(
                                [f"{method}: {count}" for method, count in method_counts.items()])
                            st.info(f"**Methods Used:** {method_summary} electronic steps")

                        st.subheader("Detailed Data")

                        df_display = df.copy()
                        df_display['Energy (eV)'] = df_display['Energy (eV)'].apply(lambda x: f"{x:.6f}")
                        df_display['Energy Difference (eV)'] = df_display['Energy Difference (eV)'].apply(
                            lambda x: f"{x:.2E}" if x != float('inf') else "N/A"
                        )

                        st.dataframe(df_display, use_container_width=True)

                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="Download Convergence Data (CSV)",
                            data=csv_data,
                            file_name="vasp_convergence_data.csv",
                            mime="text/csv",
                            type = 'primary'
                        )

                except Exception as e:
                    st.error(f"Error processing OSZICAR file: {str(e)}")
                    st.error("Please ensure your file is in the correct VASP format.")

            tab_idx += 1

        if doscar_found:
            with tabs[tab_idx]:
                st.header("Density of States (DOS) Analysis")

                try:
                    dos_df, efermi, spin_polarized, natoms = parse_doscar(doscar_content)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Fermi Energy:** {efermi:.4f} eV")
                    with col2:
                        st.info(f"**Number of Atoms:** {natoms}")
                    with col3:
                        st.info(f"**Spin Polarized:** {'Yes' if spin_polarized else 'No'}")

                    st.subheader("Electronic Structure Analysis")

                    if spin_polarized:
                        dos_at_fermi_up = dos_df.iloc[(dos_df['Energy'] - efermi).abs().argsort()[0]]['DOS_up']
                        dos_at_fermi_down = dos_df.iloc[(dos_df['Energy'] - efermi).abs().argsort()[0]]['DOS_down']
                        dos_at_fermi_total = dos_at_fermi_up + dos_at_fermi_down

                        total_electrons_up = dos_df[dos_df['Energy'] <= efermi]['DOS_up'].sum() * (
                                    dos_df['Energy'].iloc[1] - dos_df['Energy'].iloc[0])
                        total_electrons_down = dos_df[dos_df['Energy'] <= efermi]['DOS_down'].sum() * (
                                    dos_df['Energy'].iloc[1] - dos_df['Energy'].iloc[0])

                        magnetic_moment = total_electrons_up - total_electrons_down

                        col_a1, col_a2, col_a3 = st.columns(3)
                        with col_a1:
                            st.metric("DOS at E_F (total)", f"{dos_at_fermi_total:.3f} states/eV")
                        with col_a2:
                            st.metric("DOS at E_F (spin-up)", f"{dos_at_fermi_up:.3f} states/eV")
                        with col_a3:
                            st.metric("DOS at E_F (spin-down)", f"{dos_at_fermi_down:.3f} states/eV")

                        st.info(f"**Magnetic Moment:** ~{magnetic_moment:.2f} μB (approximate from DOS integration)")

                        if dos_at_fermi_total < 0.1:
                            st.success("Low DOS at Fermi level suggests semiconducting or insulating behavior")
                        else:
                            st.info("Significant DOS at Fermi level indicates metallic character")

                    else:
                        dos_at_fermi = dos_df.iloc[(dos_df['Energy'] - efermi).abs().argsort()[0]]['DOS']

                        st.metric("DOS at Fermi Level", f"{dos_at_fermi:.3f} states/eV")

                        if dos_at_fermi < 0.1:
                            st.success("Low DOS at Fermi level suggests semiconducting or insulating behavior")

                            occupied_states = dos_df[dos_df['Energy'] < efermi]
                            unoccupied_states = dos_df[dos_df['Energy'] > efermi]

                            if len(occupied_states) > 0 and len(unoccupied_states) > 0:
                                threshold = 0.01 * dos_df['DOS'].max()


                                vbm_candidates = occupied_states[occupied_states['DOS'] > threshold]
                                if len(vbm_candidates) > 0:
                                    vbm_energy = vbm_candidates['Energy'].max()
                                else:
                                    vbm_energy = None

                                cbm_candidates = unoccupied_states[unoccupied_states['DOS'] > threshold]
                                if len(cbm_candidates) > 0:
                                    cbm_energy = cbm_candidates['Energy'].min()
                                else:
                                    cbm_energy = None

                                if vbm_energy is not None and cbm_energy is not None:
                                    band_gap = cbm_energy - vbm_energy
                                    st.warning(f"**Estimated Band Gap:** ~{band_gap:.3f} eV (approximate from DOS)")
                                    st.caption(
                                        "Note: This is a rough estimate. For accurate band gaps, use band structure calculations.")
                        else:
                            st.info("Significant DOS at Fermi level indicates metallic character")

                    st.subheader("DOS Plot Options")

                    col_opt1, col_opt2 = st.columns(2)
                    with col_opt1:
                        shift_fermi = st.checkbox("Shift Fermi level to 0 eV", value=True)
                        show_fermi_line = st.checkbox("Show Fermi level line", value=True)
                    with col_opt2:
                        energy_range = st.slider(
                            "Energy range (eV)",
                            float(dos_df['Energy'].min()),
                            float(dos_df['Energy'].max()),
                            (float(dos_df['Energy'].min()), float(dos_df['Energy'].max()))
                        )

                    dos_df_original = dos_df.copy()

                    if shift_fermi:
                        dos_df['Energy'] = dos_df['Energy'] - efermi
                        fermi_position = 0.0
                        x_label = 'Energy - E_Fermi (eV)'
                    else:
                        fermi_position = efermi
                        x_label = 'Energy (eV)'

                    mask = (dos_df['Energy'] >= energy_range[0] - (efermi if shift_fermi else 0)) & \
                           (dos_df['Energy'] <= energy_range[1] - (efermi if shift_fermi else 0))
                    dos_df_plot = dos_df[mask]

                    fig_dos = go.Figure()

                    if spin_polarized:
                        if 'DOS_up' in dos_df_plot.columns and 'DOS_down' in dos_df_plot.columns:
                            fig_dos.add_trace(go.Scatter(
                                x=dos_df_plot['Energy'],
                                y=dos_df_plot['DOS_up'],
                                name='Spin Up',
                                mode='lines',
                                line=dict(width=2, color='blue'),
                                fill='tozeroy'
                            ))

                            fig_dos.add_trace(go.Scatter(
                                x=dos_df_plot['Energy'],
                                y=-dos_df_plot['DOS_down'],
                                name='Spin Down',
                                mode='lines',
                                line=dict(width=2, color='red'),
                                fill='tozeroy'
                            ))
                            y_label = 'DOS (states/eV)'
                    else:
                        fig_dos.add_trace(go.Scatter(
                            x=dos_df_plot['Energy'],
                            y=dos_df_plot['DOS'],
                            name='Total DOS',
                            mode='lines',
                            line=dict(width=2, color='blue'),
                            fill='tozeroy'
                        ))
                        y_label = 'DOS (states/eV)'

                    if show_fermi_line:
                        fig_dos.add_vline(
                            x=fermi_position,
                            line_dash="dash",
                            line_color="green",
                            line_width=2,
                            annotation_text="E_Fermi" if not shift_fermi else "E_F = 0",
                            annotation_position="top"
                        )

                    fig_dos.update_layout(
                        title=dict(
                            text="Density of States",
                            font=dict(size=24, color='black')
                        ),
                        xaxis=dict(
                            title=x_label,
                            title_font=dict(size=18, color='black'),
                            tickfont=dict(size=16, color='black'),
                            gridcolor='lightgray'
                        ),
                        yaxis=dict(
                            title=y_label,
                            title_font=dict(size=18, color='black'),
                            tickfont=dict(size=16, color='black'),
                            gridcolor='lightgray'
                        ),
                        legend=dict(
                            x=0.02,
                            y=0.98,
                            bgcolor='rgba(255,255,255,0.8)',
                            borderwidth=1,
                            font=dict(size=14)
                        ),
                        height=600,
                        plot_bgcolor='white',
                        font=dict(size=16, color='black'),
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_dos, use_container_width=True)

                    st.subheader("DOS Data")
                    st.dataframe(dos_df_plot.head(50), use_container_width=True)

                    csv_dos = dos_df_original.to_csv(index=False)
                    st.download_button(
                        label="Download DOS Data (CSV)",
                        data=csv_dos,
                        file_name="dos_data.csv",
                        mime="text/csv",
                        type='primary'
                    )

                except Exception as e:
                    st.error(f"Error processing DOSCAR file: {str(e)}")
                    st.error("Please ensure your DOSCAR file is in the correct VASP format.")

            tab_idx += 1

        if vasprun_found:
            with tabs[tab_idx]:
                st.header("DOS from vasprun.xml (Pymatgen)")

                try:
                    energies, dos_data, dos_down_data, efermi, spin_polarized, complete_dos = parse_vasprun_dos(
                        vasprun_content)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Fermi Energy:** {efermi:.4f} eV")
                    with col2:
                        st.info(f"**Spin Polarized:** {'Yes' if spin_polarized else 'No'}")

                    st.subheader("DOS Plot Options")
                    col_opt1, col_opt2 = st.columns(2)
                    with col_opt1:
                        show_fermi = st.checkbox("Show Fermi level line", value=True, key="vasprun_fermi")
                        energy_range_vasp = st.slider(
                            "Energy range (eV)",
                            float(energies.min()),
                            float(energies.max()),
                            (float(energies.min()), float(energies.max())),
                            key="vasprun_range"
                        )

                    mask = (energies >= energy_range_vasp[0]) & (energies <= energy_range_vasp[1])
                    energies_plot = energies[mask]

                    fig_dos = go.Figure()

                    if spin_polarized:
                        dos_up_plot = dos_data[mask]
                        dos_down_plot = dos_down_data[mask]

                        fig_dos.add_trace(go.Scatter(
                            x=energies_plot,
                            y=dos_up_plot,
                            name='Spin Up',
                            mode='lines',
                            line=dict(width=2, color='blue'),
                            fill='tozeroy'
                        ))

                        fig_dos.add_trace(go.Scatter(
                            x=energies_plot,
                            y=-dos_down_plot,
                            name='Spin Down',
                            mode='lines',
                            line=dict(width=2, color='red'),
                            fill='tozeroy'
                        ))
                        y_label = 'DOS (states/eV)'
                    else:
                        dos_plot = dos_data[mask]

                        fig_dos.add_trace(go.Scatter(
                            x=energies_plot,
                            y=dos_plot,
                            name='Total DOS',
                            mode='lines',
                            line=dict(width=2, color='blue'),
                            fill='tozeroy'
                        ))
                        y_label = 'DOS (states/eV)'

                    if show_fermi:
                        fig_dos.add_vline(
                            x=0,
                            line_dash="dash",
                            line_color="green",
                            line_width=2,
                            annotation_text="E_F = 0",
                            annotation_position="top"
                        )

                    fig_dos.update_layout(
                        title=dict(
                            text="Density of States (from vasprun.xml)",
                            font=dict(size=24, color='black')
                        ),
                        xaxis=dict(
                            title='Energy - E_Fermi (eV)',
                            title_font=dict(size=18, color='black'),
                            tickfont=dict(size=16, color='black'),
                            gridcolor='lightgray'
                        ),
                        yaxis=dict(
                            title=y_label,
                            title_font=dict(size=18, color='black'),
                            tickfont=dict(size=16, color='black'),
                            gridcolor='lightgray'
                        ),
                        legend=dict(
                            x=0.02,
                            y=0.98,
                            bgcolor='rgba(255,255,255,0.8)',
                            borderwidth=1,
                            font=dict(size=14)
                        ),
                        height=600,
                        plot_bgcolor='white',
                        font=dict(size=16, color='black'),
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_dos, use_container_width=True)

                    st.subheader("Electronic Structure Information")

                    fermi_idx = np.argmin(np.abs(energies))
                    if spin_polarized:
                        dos_at_fermi = dos_data[fermi_idx] + dos_down_data[fermi_idx]
                        total_dos = dos_data + dos_down_data
                    else:
                        dos_at_fermi = dos_data[fermi_idx]
                        total_dos = dos_data

                    col_dos1, col_dos2 = st.columns(2)
                    with col_dos1:
                        if spin_polarized:
                            st.metric("DOS at Fermi Level (total)", f"{dos_at_fermi:.3f} states/eV")
                        else:
                            st.metric("DOS at Fermi Level", f"{dos_at_fermi:.3f} states/eV")

                    if dos_at_fermi < 0.1:
                        occupied = energies < 0
                        if np.any(occupied):
                            occupied_dos = total_dos[occupied]
                            occupied_energies = energies[occupied]
                            threshold = 0.01 * np.max(total_dos)
                            vbm_candidates = occupied_energies[occupied_dos > threshold]
                            if len(vbm_candidates) > 0:
                                vbm_energy = np.max(vbm_candidates)
                            else:
                                vbm_energy = None
                        else:
                            vbm_energy = None

                        unoccupied = energies > 0
                        if np.any(unoccupied):
                            unoccupied_dos = total_dos[unoccupied]
                            unoccupied_energies = energies[unoccupied]
                            threshold = 0.01 * np.max(total_dos)
                            cbm_candidates = unoccupied_energies[unoccupied_dos > threshold]
                            if len(cbm_candidates) > 0:
                                cbm_energy = np.min(cbm_candidates)
                            else:
                                cbm_energy = None
                        else:
                            cbm_energy = None

                        if vbm_energy is not None and cbm_energy is not None:
                            band_gap_dos = cbm_energy - vbm_energy
                            with col_dos2:
                                st.metric("Estimated Band Gap (from DOS)", f"{band_gap_dos:.3f} eV")
                            st.success(f"Low DOS at Fermi level suggests semiconducting/insulating behavior")
                            st.caption(
                                "Note: DOS-based band gap is approximate. Use band structure for accurate values.")
                        else:
                            st.success("Low DOS at Fermi level suggests semiconducting/insulating behavior")
                    else:
                        st.info("Significant DOS at Fermi level indicates metallic character")

                    dos_df = pd.DataFrame({
                        'Energy (eV)': energies,
                        'DOS_up' if spin_polarized else 'DOS': dos_data,
                    })
                    if spin_polarized:
                        dos_df['DOS_down'] = dos_down_data

                    csv_dos = dos_df.to_csv(index=False)
                    st.download_button(
                        label="Download DOS Data (CSV)",
                        data=csv_dos,
                        file_name="vasprun_dos_data.csv",
                        mime="text/csv",
                        type='primary'
                    )

                except Exception as e:
                    st.error(f"Error processing vasprun.xml: {str(e)}")
                    st.error("Probably there is some error with the input file, maybe the VASP calculation did not finished correctly?")

            tab_idx += 1
        if vasprun_found:
            with tabs[tab_idx]:
                st.header("Band Structure from vasprun.xml (Pymatgen)")

                try:
                    from pymatgen.electronic_structure.core import Spin

                    if kpoints_found:
                        st.success("Using KPOINTS file for band structure path")
                        band_structure, efermi = parse_vasprun_bands(vasprun_content, kpoints_content)
                    else:
                        st.warning("No KPOINTS file provided. Attempting to extract from vasprun.xml...")
                        band_structure, efermi = parse_vasprun_bands(vasprun_content)

                    num_kpoints = len(band_structure.kpoints)

                    if num_kpoints <= 1:
                        st.error("**Cannot plot band structure: Only 1 k-point found (Gamma-point only calculation)**")
                        st.info("""
                                Band structure visualization requires multiple k-points along a path. 

                                Your calculation appears to use only the Gamma point, which is suitable for:
                                - DOS calculations
                                - Total energy calculations
                                - Property calculations

                                To visualize band structure, you need to:
                                1. Create a KPOINTS file with a k-point path (e.g., using a band structure generator)
                                2. Run VASP with this KPOINTS file
                                3. Upload the resulting vasprun.xml
                                """)

                        st.metric("Number of k-points", num_kpoints)
                        st.metric("Fermi Energy", f"{efermi:.4f} eV")

                    elif num_kpoints < 10:
                        st.warning(
                            f"**Warning: Only {num_kpoints} k-points found. Band structure may not be well-resolved.**")
                        st.info(
                            "For a smooth band structure plot, typically 50-100 k-points are recommended along the path.")

                    if num_kpoints > 1:
                        if efermi == 0.0:
                            try:
                                from pymatgen.io.vasp.outputs import Vasprun
                                import tempfile
                                import os

                                with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                                    f.write(vasprun_content)
                                    temp_path = f.name

                                vasprun_temp = Vasprun(temp_path, parse_dos=True, parse_eigen=False)
                                efermi = vasprun_temp.efermi
                                os.unlink(temp_path)
                            except:
                                pass

                        st.info(f"**Fermi Energy:** {efermi:.4f} eV")
                        st.info(f"**Number of k-points:** {num_kpoints}")

                        st.subheader("Band Structure Plot Options")
                        col_opt1, col_opt2 = st.columns(2)
                        with col_opt1:
                            shift_fermi_vasprun = st.checkbox("Shift Fermi level to 0 eV", value=True,
                                                              key="shift_fermi_vasprun")
                            show_fermi_vasprun = st.checkbox("Show Fermi level line", value=True,
                                                             key="show_fermi_vasprun")
                        with col_opt2:
                            energy_window_vasprun = st.slider(
                                "Energy window around Fermi (eV)",
                                1.0, 20.0, 10.0, 0.5,
                                key="energy_window_vasprun"
                            )

                        fig_bands = go.Figure()

                        for spin in band_structure.bands:
                            bands = band_structure.bands[spin]

                            distances = []
                            current_distance = 0

                            for i, kpoint in enumerate(band_structure.kpoints):
                                if i == 0:
                                    distances.append(0)
                                else:
                                    distance = np.linalg.norm(
                                        kpoint.cart_coords - band_structure.kpoints[i - 1].cart_coords
                                    )
                                    current_distance += distance
                                    distances.append(current_distance)

                            distances = np.array(distances)

                            for band_idx in range(bands.shape[0]):
                                band_energies = bands[band_idx, :]

                                if shift_fermi_vasprun:
                                    band_energies = band_energies - efermi
                                    if np.min(band_energies) > energy_window_vasprun or np.max(
                                            band_energies) < -energy_window_vasprun:
                                        continue

                                color = 'blue' if spin == Spin.up else 'red'
                                showlegend = False
                                if band_idx == 0:
                                    name = 'Spin Up' if spin == Spin.up else 'Spin Down'
                                    showlegend = True
                                else:
                                    name = None

                                fig_bands.add_trace(go.Scatter(
                                    x=distances,
                                    y=band_energies,
                                    mode='lines+markers' if num_kpoints < 20 else 'lines',
                                    line=dict(color=color, width=1),
                                    marker=dict(size=4) if num_kpoints < 20 else None,
                                    showlegend=showlegend,
                                    name=name,
                                    hovertemplate=f'Band {band_idx + 1}<br>Distance: %{{x:.3f}}<br>Energy: %{{y:.3f}} eV<extra></extra>'
                                ))

                        if show_fermi_vasprun:
                            if shift_fermi_vasprun:
                                fermi_level = 0.0
                                annotation = "E_F = 0"
                            else:
                                fermi_level = efermi
                                annotation = f"E_F = {efermi:.3f} eV"

                            fig_bands.add_hline(
                                y=fermi_level,
                                line_dash="dash",
                                line_color="green",
                                line_width=2,
                                annotation_text=annotation,
                                annotation_position="right"
                            )

                        if hasattr(band_structure, 'get_branch_indices'):
                            try:
                                branch_indices = band_structure.get_branch_indices()
                                for idx in branch_indices:
                                    if idx < len(distances):
                                        fig_bands.add_vline(
                                            x=distances[idx],
                                            line_dash="dot",
                                            line_color="gray",
                                            line_width=1
                                        )
                            except:
                                pass

                        y_label = 'Energy - E_Fermi (eV)' if shift_fermi_vasprun else 'Energy (eV)'
                        y_range = [-energy_window_vasprun, energy_window_vasprun] if shift_fermi_vasprun else None

                        fig_bands.update_layout(
                            title=dict(
                                text="Electronic Band Structure (from vasprun.xml)",
                                font=dict(size=24, color='black')
                            ),
                            xaxis=dict(
                                title='k-path',
                                title_font=dict(size=18, color='black'),
                                tickfont=dict(size=16, color='black'),
                                gridcolor='lightgray'
                            ),
                            yaxis=dict(
                                title=y_label,
                                title_font=dict(size=18, color='black'),
                                tickfont=dict(size=16, color='black'),
                                gridcolor='lightgray',
                                range=y_range
                            ),
                            width = 800,
                            height=600,
                            plot_bgcolor='white',
                            font=dict(size=16, color='black'),
                            hovermode='closest'
                        )

                        st.plotly_chart(fig_bands, use_container_width=False)

                        st.subheader("Band Gap Analysis")

                        all_bands = band_structure.bands[Spin.up]

                        if shift_fermi_vasprun:
                            bands_for_gap = all_bands - efermi
                            occupied_bands = bands_for_gap[bands_for_gap < 0]
                            unoccupied_bands = bands_for_gap[bands_for_gap > 0]
                        else:
                            bands_for_gap = all_bands
                            occupied_bands = bands_for_gap[bands_for_gap < efermi]
                            unoccupied_bands = bands_for_gap[bands_for_gap > efermi]

                        if len(occupied_bands) > 0 and len(unoccupied_bands) > 0:
                            vbm = np.max(occupied_bands)
                            cbm = np.min(unoccupied_bands)
                            band_gap = cbm - vbm

                            col_gap1, col_gap2, col_gap3 = st.columns(3)
                            with col_gap1:
                                st.metric("Valence Band Maximum (VBM)", f"{vbm:.4f} eV")
                            with col_gap2:
                                st.metric("Conduction Band Minimum (CBM)", f"{cbm:.4f} eV")
                            with col_gap3:
                                st.metric("Band Gap", f"{band_gap:.4f} eV")

                            if band_gap < 0.1:
                                st.info("Material appears to be metallic (no band gap)")
                            elif band_gap < 2.0:
                                st.success(f"Material is a semiconductor with band gap of {band_gap:.4f} eV")
                            else:
                                st.success(f"Material is an insulator with band gap of {band_gap:.4f} eV")
                        else:
                            st.info("Cannot determine band gap - all bands may be partially occupied (metallic)")

                except Exception as e:
                    st.error(f"Error processing band structure: {str(e)}")
                    st.error("Make sure your vasprun.xml contains band structure data")

            tab_idx += 1
        if vasprun_found:
            with tabs[tab_idx]:
                st.header("Ionic Relaxation from vasprun.xml")

                try:
                    ionic_data, final_structure, initial_structure = parse_vasprun_ionic_steps(vasprun_content)

                    num_steps = len(ionic_data['energies'])

                    if num_steps <= 1:
                        st.warning("This appears to be a single-point calculation (no ionic relaxation).")
                        st.info(f"**Total Energy:** {ionic_data['energies'][0]:.6f} eV")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.info(f"**Ionic Steps:** {num_steps}")
                        with col2:
                            st.info(f"**Initial Energy:** {ionic_data['energies'][0]:.6f} eV")
                        with col3:
                            st.info(f"**Final Energy:** {ionic_data['energies'][-1]:.6f} eV")

                        st.subheader("Energy Convergence")

                        steps = list(range(1, num_steps + 1))
                        energies = ionic_data['energies']

                        energy_diffs = [float('inf')]
                        for i in range(1, len(energies)):
                            energy_diffs.append(abs(energies[i] - energies[i - 1]))

                        fig_energy = go.Figure()

                        fig_energy.add_trace(go.Scatter(
                            x=steps,
                            y=energies,
                            name='Total Energy',
                            mode='lines+markers',
                            line=dict(width=3, color='blue'),
                            marker=dict(size=8),
                            yaxis='y1',
                            hovertemplate='Step: %{x}<br>Energy: %{y:.6f} eV<extra></extra>'
                        ))

                        if len(energy_diffs) > 1:
                            valid_diffs = [diff for diff in energy_diffs[1:] if diff != float('inf')]
                            if valid_diffs:
                                fig_energy.add_trace(go.Scatter(
                                    x=steps[1:],
                                    y=valid_diffs,
                                    name='|ΔE|',
                                    mode='lines+markers',
                                    line=dict(width=2, color='red', dash='dash'),
                                    marker=dict(size=6, symbol='square'),
                                    yaxis='y2',
                                    hovertemplate='Step: %{x}<br>|ΔE|: %{y:.2E} eV<extra></extra>'
                                ))

                        fig_energy.update_layout(
                            title=dict(
                                text="Energy Convergence",
                                font=dict(size=24, color='black')
                            ),
                            xaxis=dict(
                                title='Ionic Step',
                                title_font=dict(size=20, color='black'),
                                tickfont=dict(size=18, color='black'),
                                gridcolor='lightgray'
                            ),
                            yaxis=dict(
                                title='Total Energy (eV)',
                                side='left',
                                title_font=dict(size=20, color='blue'),
                                tickfont=dict(size=18, color='blue')
                            ),
                            yaxis2=dict(
                                title='|Energy Difference| (eV)',
                                side='right',
                                overlaying='y',
                                title_font=dict(size=20, color='red'),
                                tickfont=dict(size=18, color='red'),
                                tickformat='.1E'
                            ),
                            legend=dict(
                                x=0.5,
                                y=-0.2,
                                xanchor='center',
                                orientation='h',
                                bgcolor='rgba(255,255,255,0.8)',
                                borderwidth=1,
                                font=dict(size=18)
                            ),
                            height=600,
                            plot_bgcolor='white',
                            font=dict(size=18, color='black')
                        )

                        st.plotly_chart(fig_energy, use_container_width=True)

                        if ionic_data['forces']:
                            st.subheader("Forces Convergence")

                            fig_forces = go.Figure()

                            fig_forces.add_trace(go.Scatter(
                                x=steps,
                                y=ionic_data['forces'],
                                name='Max Force',
                                mode='lines+markers',
                                line=dict(width=3, color='green'),
                                marker=dict(size=8),
                                hovertemplate='Step: %{x}<br>Max Force: %{y:.4f} eV/Å<extra></extra>'
                            ))

                            fig_forces.update_layout(
                                title=dict(
                                    text="Maximum Force per Ionic Step",
                                    font=dict(size=24, color='black')
                                ),
                                xaxis=dict(
                                    title='Ionic Step',
                                    title_font=dict(size=20, color='black'),
                                    tickfont=dict(size=18, color='black'),
                                    gridcolor='lightgray'
                                ),
                                yaxis=dict(
                                    title='Maximum Force (eV/Å)',
                                    title_font=dict(size=20, color='green'),
                                    tickfont=dict(size=18, color='green')
                                ),
                                height=500,
                                plot_bgcolor='white',
                                font=dict(size=18, color='black')
                            )

                            st.plotly_chart(fig_forces, use_container_width=True)

                        if ionic_data['structures']:
                            st.subheader("Lattice Evolution")

                            lattice_a = []
                            lattice_b = []
                            lattice_c = []
                            volumes = []

                            for structure in ionic_data['structures']:
                                lattice = structure.lattice
                                lattice_a.append(lattice.a)
                                lattice_b.append(lattice.b)
                                lattice_c.append(lattice.c)
                                volumes.append(lattice.volume)

                            fig_lattice = go.Figure()

                            fig_lattice.add_trace(go.Scatter(
                                x=steps,
                                y=lattice_a,
                                name='a',
                                mode='lines+markers',
                                line=dict(width=2),
                                marker=dict(size=6)
                            ))

                            fig_lattice.add_trace(go.Scatter(
                                x=steps,
                                y=lattice_b,
                                name='b',
                                mode='lines+markers',
                                line=dict(width=2),
                                marker=dict(size=6)
                            ))

                            fig_lattice.add_trace(go.Scatter(
                                x=steps,
                                y=lattice_c,
                                name='c',
                                mode='lines+markers',
                                line=dict(width=2),
                                marker=dict(size=6)
                            ))

                            fig_lattice.update_layout(
                                title=dict(
                                    text="Lattice Parameters Evolution",
                                    font=dict(size=24, color='black')
                                ),
                                xaxis=dict(
                                    title='Ionic Step',
                                    title_font=dict(size=20, color='black'),
                                    tickfont=dict(size=18, color='black'),
                                    gridcolor='lightgray'
                                ),
                                yaxis=dict(
                                    title='Lattice Parameter (Å)',
                                    title_font=dict(size=20, color='black'),
                                    tickfont=dict(size=18, color='black')
                                ),
                                legend=dict(
                                    x=0.02,
                                    y=0.98,
                                    bgcolor='rgba(255,255,255,0.8)',
                                    borderwidth=1,
                                    font=dict(size=16)
                                ),
                                height=500,
                                plot_bgcolor='white',
                                font=dict(size=18, color='black')
                            )

                            st.plotly_chart(fig_lattice, use_container_width=True)

                            fig_volume = go.Figure()

                            fig_volume.add_trace(go.Scatter(
                                x=steps,
                                y=volumes,
                                name='Volume',
                                mode='lines+markers',
                                line=dict(width=3, color='purple'),
                                marker=dict(size=8),
                                hovertemplate='Step: %{x}<br>Volume: %{y:.2f} Ų<extra></extra>'
                            ))

                            fig_volume.update_layout(
                                title=dict(
                                    text="Cell Volume Evolution",
                                    font=dict(size=24, color='black')
                                ),
                                xaxis=dict(
                                    title='Ionic Step',
                                    title_font=dict(size=20, color='black'),
                                    tickfont=dict(size=18, color='black'),
                                    gridcolor='lightgray'
                                ),
                                yaxis=dict(
                                    title='Volume (ų)',
                                    title_font=dict(size=20, color='purple'),
                                    tickfont=dict(size=18, color='purple')
                                ),
                                height=500,
                                plot_bgcolor='white',
                                font=dict(size=18, color='black')
                            )

                            st.plotly_chart(fig_volume, use_container_width=True)

                        st.subheader("Summary")

                        energy_change = energies[-1] - energies[0]
                        final_max_force = ionic_data['forces'][-1] if ionic_data['forces'] else None

                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        with col_sum1:
                            st.metric("Energy Change", f"{energy_change:.6f} eV")
                        with col_sum2:
                            if final_max_force is not None:
                                st.metric("Final Max Force", f"{final_max_force:.4f} eV/Å")
                        with col_sum3:
                            if volumes:
                                volume_change = ((volumes[-1] - volumes[0]) / volumes[0]) * 100
                                st.metric("Volume Change", f"{volume_change:.2f} %")

                        if ionic_data['structures']:
                            st.subheader("Lattice Parameters Summary")

                            initial_struct = ionic_data['structures'][0]
                            final_struct = ionic_data['structures'][-1]

                            col_lat1, col_lat2, col_lat3 = st.columns(3)

                            with col_lat1:
                                st.markdown("**Initial Lattice**")
                                st.write(f"a = {initial_struct.lattice.a:.4f} Å")
                                st.write(f"b = {initial_struct.lattice.b:.4f} Å")
                                st.write(f"c = {initial_struct.lattice.c:.4f} Å")
                                st.write(f"α = {initial_struct.lattice.alpha:.2f}°")
                                st.write(f"β = {initial_struct.lattice.beta:.2f}°")
                                st.write(f"γ = {initial_struct.lattice.gamma:.2f}°")
                                st.write(f"Volume = {initial_struct.lattice.volume:.2f} ų")

                            with col_lat2:
                                st.markdown("**Final Lattice**")
                                st.write(f"a = {final_struct.lattice.a:.4f} Å")
                                st.write(f"b = {final_struct.lattice.b:.4f} Å")
                                st.write(f"c = {final_struct.lattice.c:.4f} Å")
                                st.write(f"α = {final_struct.lattice.alpha:.2f}°")
                                st.write(f"β = {final_struct.lattice.beta:.2f}°")
                                st.write(f"γ = {final_struct.lattice.gamma:.2f}°")
                                st.write(f"Volume = {final_struct.lattice.volume:.2f} ų")

                            with col_lat3:
                                st.markdown("**Change (%)**")
                                a_change = ((
                                                        final_struct.lattice.a - initial_struct.lattice.a) / initial_struct.lattice.a) * 100
                                b_change = ((
                                                        final_struct.lattice.b - initial_struct.lattice.b) / initial_struct.lattice.b) * 100
                                c_change = ((
                                                        final_struct.lattice.c - initial_struct.lattice.c) / initial_struct.lattice.c) * 100
                                alpha_change = final_struct.lattice.alpha - initial_struct.lattice.alpha
                                beta_change = final_struct.lattice.beta - initial_struct.lattice.beta
                                gamma_change = final_struct.lattice.gamma - initial_struct.lattice.gamma
                                vol_change = ((
                                                          final_struct.lattice.volume - initial_struct.lattice.volume) / initial_struct.lattice.volume) * 100

                                st.write(f"Δa = {a_change:+.2f}%")
                                st.write(f"Δb = {b_change:+.2f}%")
                                st.write(f"Δc = {c_change:+.2f}%")
                                st.write(f"Δα = {alpha_change:+.2f}°")
                                st.write(f"Δβ = {beta_change:+.2f}°")
                                st.write(f"Δγ = {gamma_change:+.2f}°")
                                st.write(f"ΔVolume = {vol_change:+.2f}%")

                        convergence_data = pd.DataFrame({
                            'Ionic Step': steps,
                            'Energy (eV)': energies,
                            'Energy Diff (eV)': energy_diffs,
                            'Max Force (eV/Å)': ionic_data['forces'] if ionic_data['forces'] else [None] * num_steps
                        })

                        if ionic_data['structures']:
                            lattice_alpha = []
                            lattice_beta = []
                            lattice_gamma = []

                            for structure in ionic_data['structures']:
                                lattice = structure.lattice
                                lattice_alpha.append(lattice.alpha)
                                lattice_beta.append(lattice.beta)
                                lattice_gamma.append(lattice.gamma)

                            convergence_data['a (Å)'] = lattice_a
                            convergence_data['b (Å)'] = lattice_b
                            convergence_data['c (Å)'] = lattice_c
                            convergence_data['α (°)'] = lattice_alpha
                            convergence_data['β (°)'] = lattice_beta
                            convergence_data['γ (°)'] = lattice_gamma
                            convergence_data['Volume (ų)'] = volumes

                        st.subheader("Detailed Convergence Data")
                        st.dataframe(convergence_data, use_container_width=True)

                        csv_conv = convergence_data.to_csv(index=False)
                        st.download_button(
                            label="Download Convergence Data (CSV)",
                            data=csv_conv,
                            file_name="ionic_convergence.csv",
                            mime="text/csv",
                            type='primary'
                        )

                except Exception as e:
                    st.error(f"Error processing ionic steps: {str(e)}")
                    import traceback

                    st.error(traceback.format_exc())

            tab_idx += 1
        if eigenval_found:
            with tabs[tab_idx]:
                st.header("Electronic Band Structure")

                try:
                    kpoints, eigenvalues, k_distances, nbands, nkpts = parse_eigenval(eigenval_content)

                    efermi_band = 0.0
                    if doscar_found and doscar_content:
                        try:
                            _, efermi_band, _, _ = parse_doscar(doscar_content)
                            st.info(f"Using Fermi energy from DOSCAR: {efermi_band:.4f} eV")
                        except:
                            st.warning("Could not extract Fermi energy from DOSCAR. Using 0 eV as reference.")
                    else:
                        st.warning("No DOSCAR file found. Using 0 eV as Fermi energy reference.")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Number of k-points:** {nkpts}")
                    with col2:
                        st.info(f"**Number of bands:** {nbands}")
                    with col3:
                        st.info(f"**Fermi Energy:** {efermi_band:.4f} eV")

                    st.subheader("Band Structure Plot Options")

                    col_opt1, col_opt2 = st.columns(2)
                    with col_opt1:
                        shift_fermi_band = st.checkbox("Shift Fermi level to 0 eV", value=True, key="shift_fermi_band")
                        show_fermi_band = st.checkbox("Show Fermi level line", value=True, key="show_fermi_band")
                    with col_opt2:
                        energy_window = st.slider(
                            "Energy window around Fermi (eV)",
                            1.0, 20.0, 10.0, 0.5
                        )

                    eigenvalues_plot = eigenvalues.copy()
                    if shift_fermi_band:
                        eigenvalues_plot = eigenvalues_plot - efermi_band
                        fermi_level = 0.0
                        y_label = 'Energy - E_Fermi (eV)'
                    else:
                        fermi_level = efermi_band
                        y_label = 'Energy (eV)'

                    fig_bands = go.Figure()

                    for iband in range(nbands):
                        band_energies = eigenvalues_plot[:, iband]

                        if shift_fermi_band:
                            if np.min(band_energies) > energy_window or np.max(band_energies) < -energy_window:
                                continue

                        fig_bands.add_trace(go.Scatter(
                            x=k_distances,
                            y=band_energies,
                            mode='lines',
                            line=dict(color='blue', width=1),
                            showlegend=False,
                            hovertemplate=f'Band {iband + 1}<br>k-distance: %{{x:.3f}}<br>Energy: %{{y:.3f}} eV<extra></extra>'
                        ))

                    if show_fermi_band:
                        fig_bands.add_hline(
                            y=fermi_level,
                            line_dash="dash",
                            line_color="red",
                            line_width=2,
                            annotation_text="E_Fermi",
                            annotation_position="right"
                        )

                    fig_bands.update_layout(
                        title=dict(
                            text="Electronic Band Structure (from vasprun.xml)",
                            font=dict(size=24, color='black')
                        ),
                        xaxis=dict(
                            title='k-path',
                            title_font=dict(size=18, color='black'),
                            tickfont=dict(size=16, color='black'),
                            gridcolor='lightgray'
                        ),
                        yaxis=dict(
                            title=y_label,
                            title_font=dict(size=18, color='black'),
                            tickfont=dict(size=16, color='black'),
                            gridcolor='lightgray',
                            range=[-energy_window, energy_window] if shift_fermi_band else None
                        ),
                        width=800,
                        height=600,
                        plot_bgcolor='white',
                        font=dict(size=16, color='black'),
                        hovermode='closest'
                    )

                    st.plotly_chart(fig_bands, use_container_width=False)

                    st.subheader("Band Gap Analysis")

                    if shift_fermi_band:
                        occupied_bands = eigenvalues_plot[eigenvalues_plot < 0]
                        unoccupied_bands = eigenvalues_plot[eigenvalues_plot > 0]

                        if len(occupied_bands) > 0 and len(unoccupied_bands) > 0:
                            vbm = np.max(occupied_bands)
                            cbm = np.min(unoccupied_bands)
                            band_gap = cbm - vbm

                            col_gap1, col_gap2, col_gap3 = st.columns(3)
                            with col_gap1:
                                st.metric("Valence Band Maximum (VBM)", f"{vbm:.4f} eV")
                            with col_gap2:
                                st.metric("Conduction Band Minimum (CBM)", f"{cbm:.4f} eV")
                            with col_gap3:
                                st.metric("Band Gap", f"{band_gap:.4f} eV")

                            if band_gap < 0.1:
                                st.info("Material appears to be metallic (no band gap)")
                            elif band_gap < 2.0:
                                st.success(f"Material is a semiconductor with band gap of {band_gap:.4f} eV")
                            else:
                                st.success(f"Material is an insulator with band gap of {band_gap:.4f} eV")
                        else:
                            st.info("Cannot determine band gap - all bands may be partially occupied (metallic)")
                    else:
                        st.info("Enable 'Shift Fermi level to 0 eV' for automatic band gap analysis")

                    st.subheader("Band Structure Data")

                    band_data_sample = []
                    for ik in range(min(10, nkpts)):
                        for ib in range(min(5, nbands)):
                            band_data_sample.append({
                                'k-point': ik + 1,
                                'Band': ib + 1,
                                'Energy (eV)': eigenvalues[ik, ib]
                            })

                    df_bands = pd.DataFrame(band_data_sample)
                    st.dataframe(df_bands, use_container_width=True)
                    st.caption("Showing first 10 k-points and first 5 bands")

                    csv_bands = pd.DataFrame({
                        'k_distance': k_distances,
                        **{f'band_{i + 1}': eigenvalues[:, i] for i in range(nbands)}
                    })

                    st.download_button(
                        label="Download Band Structure Data (CSV)",
                        data=csv_bands.to_csv(index=False),
                        file_name="band_structure.csv",
                        mime="text/csv",
                        type='primary'

                    )

                except Exception as e:
                    st.error(f"Error processing EIGENVAL file: {str(e)}")
                    st.error("Please ensure your EIGENVAL file is in the correct VASP format.")

            tab_idx += 1
        if poscar_found or contcar_found:
            with tabs[tab_idx]:
                st.header("Structure Viewer")

                try:
                    from pymatgen.io.vasp import Poscar
                    from pymatgen.io.ase import AseAtomsAdaptor
                    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                    import py3Dmol
                    from io import StringIO
                    from ase.io import write

                    structures = {}
                    space_groups = {}

                    if poscar_found:
                        structures['POSCAR'] = Poscar.from_str(poscar_content).structure
                        try:
                            analyzer = SpacegroupAnalyzer(structures['POSCAR'], symprec=0.1)
                            space_groups['POSCAR'] = {
                                'symbol': analyzer.get_space_group_symbol(),
                                'number': analyzer.get_space_group_number()
                            }
                        except:
                            space_groups['POSCAR'] = None

                    if contcar_found:
                        structures['CONTCAR'] = Poscar.from_str(contcar_content).structure
                        try:
                            analyzer = SpacegroupAnalyzer(structures['CONTCAR'], symprec=0.1)
                            space_groups['CONTCAR'] = {
                                'symbol': analyzer.get_space_group_symbol(),
                                'number': analyzer.get_space_group_number()
                            }
                        except:
                            space_groups['CONTCAR'] = None

                    if len(structures) == 2:
                        st.subheader("Structure Comparison: POSCAR vs CONTCAR")

                        poscar_struct = structures['POSCAR']
                        contcar_struct = structures['CONTCAR']

                        col_info1, col_info2 = st.columns(2)

                        with col_info1:
                            st.markdown("**POSCAR (Initial)**")
                            sg_text = ""
                            if space_groups.get('POSCAR'):
                                sg = space_groups['POSCAR']
                                sg_text = f" | Space Group: {sg['symbol']} (#{sg['number']})"

                            info_text = f"**Formula:** {poscar_struct.composition.reduced_formula} | **Atoms:** {len(poscar_struct)} | **Volume:** {poscar_struct.lattice.volume:.2f} Å³{sg_text}"
                            st.info(info_text)

                        with col_info2:
                            st.markdown("**CONTCAR (Final)**")
                            sg_text = ""
                            if space_groups.get('CONTCAR'):
                                sg = space_groups['CONTCAR']
                                sg_text = f" | Space Group: {sg['symbol']} (#{sg['number']})"

                            info_text = f"**Formula:** {contcar_struct.composition.reduced_formula} | **Atoms:** {len(contcar_struct)} | **Volume:** {contcar_struct.lattice.volume:.2f} Å³{sg_text}"
                            st.info(info_text)

                        st.subheader("Lattice Parameters Comparison")

                        lattice_data = {
                            'Parameter': ['a (Å)', 'b (Å)', 'c (Å)', 'α (°)', 'β (°)', 'γ (°)', 'Volume (Å³)'],
                            'POSCAR': [
                                f"{poscar_struct.lattice.a:.4f}",
                                f"{poscar_struct.lattice.b:.4f}",
                                f"{poscar_struct.lattice.c:.4f}",
                                f"{poscar_struct.lattice.alpha:.2f}",
                                f"{poscar_struct.lattice.beta:.2f}",
                                f"{poscar_struct.lattice.gamma:.2f}",
                                f"{poscar_struct.lattice.volume:.2f}"
                            ],
                            'CONTCAR': [
                                f"{contcar_struct.lattice.a:.4f}",
                                f"{contcar_struct.lattice.b:.4f}",
                                f"{contcar_struct.lattice.c:.4f}",
                                f"{contcar_struct.lattice.alpha:.2f}",
                                f"{contcar_struct.lattice.beta:.2f}",
                                f"{contcar_struct.lattice.gamma:.2f}",
                                f"{contcar_struct.lattice.volume:.2f}"
                            ],
                            'Change': [
                                f"{(contcar_struct.lattice.a - poscar_struct.lattice.a):+.4f}",
                                f"{(contcar_struct.lattice.b - poscar_struct.lattice.b):+.4f}",
                                f"{(contcar_struct.lattice.c - poscar_struct.lattice.c):+.4f}",
                                f"{(contcar_struct.lattice.alpha - poscar_struct.lattice.alpha):+.2f}",
                                f"{(contcar_struct.lattice.beta - poscar_struct.lattice.beta):+.2f}",
                                f"{(contcar_struct.lattice.gamma - poscar_struct.lattice.gamma):+.2f}",
                                f"{(contcar_struct.lattice.volume - poscar_struct.lattice.volume):+.2f}"
                            ],
                            'Change (%)': [
                                f"{((contcar_struct.lattice.a - poscar_struct.lattice.a) / poscar_struct.lattice.a * 100):+.2f}",
                                f"{((contcar_struct.lattice.b - poscar_struct.lattice.b) / poscar_struct.lattice.b * 100):+.2f}",
                                f"{((contcar_struct.lattice.c - poscar_struct.lattice.c) / poscar_struct.lattice.c * 100):+.2f}",
                                f"-",
                                f"-",
                                f"-",
                                f"{((contcar_struct.lattice.volume - poscar_struct.lattice.volume) / poscar_struct.lattice.volume * 100):+.2f}"
                            ]
                        }

                        df_lattice = pd.DataFrame(lattice_data)
                        st.dataframe(df_lattice, use_container_width=True)

                        params_to_plot = ['a', 'b', 'c', 'Volume']
                        percent_changes = [
                            ((contcar_struct.lattice.a - poscar_struct.lattice.a) / poscar_struct.lattice.a * 100),
                            ((contcar_struct.lattice.b - poscar_struct.lattice.b) / poscar_struct.lattice.b * 100),
                            ((contcar_struct.lattice.c - poscar_struct.lattice.c) / poscar_struct.lattice.c * 100),
                            ((
                                         contcar_struct.lattice.volume - poscar_struct.lattice.volume) / poscar_struct.lattice.volume * 100)
                        ]

                        fig_percent = go.Figure()

                        fig_percent.add_trace(go.Bar(
                            x=params_to_plot,
                            y=percent_changes,
                            marker_color='steelblue',
                            text=[f"{v:+.3f}%" for v in percent_changes],
                            textposition='outside',
                            textfont=dict(size=18, color='black'),
                            hovertemplate='%{x}<br>Change: %{y:.3f}%<extra></extra>'
                        ))

                        fig_percent.add_hline(
                            y=0,
                            line_dash="dash",
                            line_color="gray",
                            line_width=2
                        )

                        fig_percent.update_layout(
                            title=dict(
                                text="Percentage Change in Lattice Parameters (POSCAR → CONTCAR)",
                                font=dict(size=24, color='black')
                            ),
                            xaxis=dict(
                                title='Parameter',
                                title_font=dict(size=20, color='black'),
                                tickfont=dict(size=18, color='black')
                            ),
                            yaxis=dict(
                                title='Change (%)',
                                title_font=dict(size=20, color='black'),
                                tickfont=dict(size=18, color='black'),
                                gridcolor='lightgray',
                                zeroline=True
                            ),
                            height=500,
                            plot_bgcolor='white',
                            font=dict(size=18),
                            showlegend=False
                        )

                        st.plotly_chart(fig_percent, use_container_width=True)

                        if len(poscar_struct) == len(contcar_struct):
                            st.subheader("Atomic Displacements")

                            max_displacement = 0
                            avg_displacement = 0
                            displacements = []

                            for i in range(len(poscar_struct)):
                                frac_pos = poscar_struct[i].frac_coords
                                frac_cont = contcar_struct[i].frac_coords

                                frac_diff = frac_cont - frac_pos
                                frac_diff = frac_diff - np.round(frac_diff)

                                cart_displacement = contcar_struct.lattice.get_cartesian_coords(frac_diff)
                                displacement = np.linalg.norm(cart_displacement)

                                displacements.append(displacement)
                                max_displacement = max(max_displacement, displacement)
                                avg_displacement += displacement

                            avg_displacement /= len(poscar_struct)

                            col_disp1, col_disp2 = st.columns(2)
                            with col_disp1:
                                st.metric("Maximum Displacement", f"{max_displacement:.4f} Å")
                            with col_disp2:
                                st.metric("Average Displacement", f"{avg_displacement:.4f} Å")

                            fig_disp = go.Figure()
                            fig_disp.add_trace(go.Scatter(
                                y=displacements,
                                mode='markers',
                                marker=dict(size=12, color=displacements, colorscale='Viridis', showscale=True,
                                            colorbar=dict(
                                                title="Displacement (Å)",
                                                title_font=dict(size=18),
                                                tickfont=dict(size=16),
                                                thickness=20,
                                                len=0.7
                                            )),
                                hovertemplate='Atom %{x}<br>Displacement: %{y:.4f} Å<extra></extra>'
                            ))

                            fig_disp.update_layout(
                                title=dict(
                                    text="Atomic Displacements (POSCAR → CONTCAR)",
                                    font=dict(size=24, color='black')
                                ),
                                xaxis=dict(
                                    title='Atom Index',
                                    title_font=dict(size=20, color='black'),
                                    tickfont=dict(size=18, color='black'),
                                    gridcolor='lightgray'
                                ),
                                yaxis=dict(
                                    title='Displacement (Å)',
                                    title_font=dict(size=20, color='black'),
                                    tickfont=dict(size=18, color='black'),
                                    gridcolor='lightgray'
                                ),
                                height=500,
                                plot_bgcolor='white',
                                font=dict(size=18, color='black')
                            )

                            st.plotly_chart(fig_disp, use_container_width=True)
                        else:
                            st.warning("Structures have different number of atoms. Cannot calculate displacements.")

                        csv_comparison = df_lattice.to_csv(index=False)
                        st.download_button(
                            label="Download Comparison Data (CSV)",
                            data=csv_comparison,
                            file_name="poscar_contcar_comparison.csv",
                            mime="text/csv",
                            type='primary'
                        )

                    else:
                        struct_name = list(structures.keys())[0]
                        struct = structures[struct_name]

                        st.subheader(f"{struct_name} Structure")

                        sg_text = ""
                        if space_groups.get(struct_name):
                            sg = space_groups[struct_name]
                            sg_text = f" | Space Group: {sg['symbol']} (#{sg['number']})"

                        info_text = f"**Formula:** {struct.composition.reduced_formula} | **Atoms:** {len(struct)} | **Volume:** {struct.lattice.volume:.2f} Å³{sg_text}"
                        st.info(info_text)

                        st.subheader("Lattice Parameters")

                        lattice_single = {
                            'Parameter': ['a', 'b', 'c', 'α', 'β', 'γ', 'Volume'],
                            'Value': [
                                f"{struct.lattice.a:.4f} Å",
                                f"{struct.lattice.b:.4f} Å",
                                f"{struct.lattice.c:.4f} Å",
                                f"{struct.lattice.alpha:.2f}°",
                                f"{struct.lattice.beta:.2f}°",
                                f"{struct.lattice.gamma:.2f}°",
                                f"{struct.lattice.volume:.2f} Å³"
                            ]
                        }

                        df_single = pd.DataFrame(lattice_single)
                        st.dataframe(df_single, width='content')

                    st.subheader("3D Structure Visualization")

                    col_vis_opt1, col_vis_opt2, col_vis_opt3 = st.columns(3)

                    with col_vis_opt1:
                        show_labels = st.checkbox("Show atom labels", value=True, key="show_atom_labels_struct")
                    with col_vis_opt2:
                        show_cell_vectors = st.checkbox("Show cell vectors", value=True, key="show_cell_vectors")
                    with col_vis_opt3:
                        atom_style = st.selectbox(
                            "Atom style",
                            ["sphere",  "cross", ],
                            index=0,
                            key="atom_style_select"
                        )

                    if atom_style == "sphere":
                        atom_radius = st.slider("Sphere radius", 0.1, 1.0, 0.4, 0.05, key="sphere_radius")
                    elif atom_style == "stick":
                        stick_radius = st.slider("Stick radius", 0.05, 0.5, 0.15, 0.05, key="stick_radius")
                    elif atom_style == "line":
                        line_width = st.slider("Line width", 1, 10, 2, 1, key="line_width")


                    def add_box(view, cell, color='black', linewidth=0.05):
                        vertices = [
                            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
                        ]
                        edges = [
                            (0, 1), (1, 2), (2, 3), (3, 0),
                            (4, 5), (5, 6), (6, 7), (7, 4),
                            (0, 4), (1, 5), (2, 6), (3, 7)
                        ]
                        cart_vertices = [np.dot(v, cell) for v in vertices]
                        for i, j in edges:
                            v1, v2 = cart_vertices[i], cart_vertices[j]
                            view.addCylinder({
                                'start': {'x': v1[0], 'y': v1[1], 'z': v1[2]},
                                'end': {'x': v2[0], 'y': v2[1], 'z': v2[2]},
                                'radius': linewidth, 'color': color, 'fromCap': 1, 'toCap': 1
                            })


                    def add_cell_vectors(view, cell):
                        origin = [0, 0, 0]
                        colors = ['red', 'green', 'blue']
                        labels = ['a', 'b', 'c']

                        for i in range(3):
                            end = cell[i]
                            view.addArrow({
                                'start': {'x': origin[0], 'y': origin[1], 'z': origin[2]},
                                'end': {'x': end[0], 'y': end[1], 'z': end[2]},
                                'radius': 0.15,
                                'color': colors[i]
                            })
                            view.addLabel(labels[i], {
                                'position': {'x': end[0] * 1.1, 'y': end[1] * 1.1, 'z': end[2] * 1.1},
                                'backgroundColor': colors[i],
                                'fontColor': 'white',
                                'fontSize': 14,
                                'borderThickness': 1
                            })


                    def apply_atom_style(view, style_type):
                        if style_type == "sphere":
                            view.setStyle({'model': 0}, {"sphere": {"radius": atom_radius, "colorscheme": "Jmol"}})
                        elif style_type == "stick":
                            view.setStyle({'model': 0}, {"stick": {"radius": stick_radius, "colorscheme": "Jmol"}})
                        elif style_type == "line":
                            view.setStyle({'model': 0}, {"line": {"linewidth": line_width, "colorscheme": "Jmol"}})
                        elif style_type == "cross":
                            view.setStyle({'model': 0}, {"cross": {"radius": 0.1, "colorscheme": "Jmol"}})
                        elif style_type == "cartoon":
                            view.setStyle({'model': 0}, {"cartoon": {"color": "spectrum"}})


                    if len(structures) == 2:
                        col_viz1, col_viz2 = st.columns(2)

                        with col_viz1:
                            st.markdown("**POSCAR (Initial)**")
                            ase_poscar = AseAtomsAdaptor.get_atoms(structures['POSCAR'])
                            xyz_io = StringIO()
                            write(xyz_io, ase_poscar, format="xyz")
                            xyz_str = xyz_io.getvalue()
                            view = py3Dmol.view(width=400, height=400)
                            view.addModel(xyz_str, "xyz")
                            apply_atom_style(view, atom_style)
                            add_box(view, ase_poscar.get_cell(), color='black', linewidth=0.05)

                            if show_cell_vectors:
                                add_cell_vectors(view, ase_poscar.get_cell())

                            if show_labels:
                                for i, atom in enumerate(ase_poscar):
                                    view.addLabel(f"{atom.symbol}{i}", {
                                        'position': {'x': atom.position[0], 'y': atom.position[1],
                                                     'z': atom.position[2]},
                                        'backgroundColor': 'white',
                                        'fontColor': 'black',
                                        'fontSize': 10,
                                        'borderThickness': 0.5,
                                        'borderColor': 'grey'
                                    })

                            view.zoomTo()
                            view.zoom(1.1)
                            html_viz = view._make_html()
                            st.components.v1.html(html_viz, height=420)

                        with col_viz2:
                            st.markdown("**CONTCAR (Final)**")
                            ase_contcar = AseAtomsAdaptor.get_atoms(structures['CONTCAR'])
                            xyz_io = StringIO()
                            write(xyz_io, ase_contcar, format="xyz")
                            xyz_str = xyz_io.getvalue()
                            view = py3Dmol.view(width=400, height=400)
                            view.addModel(xyz_str, "xyz")
                            apply_atom_style(view, atom_style)
                            add_box(view, ase_contcar.get_cell(), color='black', linewidth=0.05)

                            if show_cell_vectors:
                                add_cell_vectors(view, ase_contcar.get_cell())

                            if show_labels:
                                for i, atom in enumerate(ase_contcar):
                                    view.addLabel(f"{atom.symbol}{i}", {
                                        'position': {'x': atom.position[0], 'y': atom.position[1],
                                                     'z': atom.position[2]},
                                        'backgroundColor': 'white',
                                        'fontColor': 'black',
                                        'fontSize': 10,
                                        'borderThickness': 0.5,
                                        'borderColor': 'grey'
                                    })

                            view.zoomTo()
                            view.zoom(1.1)
                            html_viz = view._make_html()
                            st.components.v1.html(html_viz, height=420)

                        if show_labels:
                            st.subheader("Atomic Positions")

                            col_table1, col_table2 = st.columns(2)

                            with col_table1:
                                st.markdown("**POSCAR Atoms**")
                                atom_data_poscar = []
                                cell_poscar = ase_poscar.get_cell()
                                inv_cell_poscar = np.linalg.inv(cell_poscar) if np.linalg.det(
                                    cell_poscar) > 1e-9 else np.eye(3)

                                for i, atom in enumerate(ase_poscar):
                                    pos = atom.position
                                    frac = np.dot(pos, inv_cell_poscar)
                                    atom_data_poscar.append({
                                        'Atom': f"{atom.symbol}{i}",
                                        'Element': atom.symbol,
                                        'X': f"{pos[0]:.3f}",
                                        'Y': f"{pos[1]:.3f}",
                                        'Z': f"{pos[2]:.3f}",
                                        'Frac_X': f"{frac[0]:.3f}",
                                        'Frac_Y': f"{frac[1]:.3f}",
                                        'Frac_Z': f"{frac[2]:.3f}"
                                    })

                                df_poscar_atoms = pd.DataFrame(atom_data_poscar)
                                st.dataframe(df_poscar_atoms, height=300, use_container_width=True)

                            with col_table2:
                                st.markdown("**CONTCAR Atoms**")
                                atom_data_contcar = []
                                cell_contcar = ase_contcar.get_cell()
                                inv_cell_contcar = np.linalg.inv(cell_contcar) if np.linalg.det(
                                    cell_contcar) > 1e-9 else np.eye(3)

                                for i, atom in enumerate(ase_contcar):
                                    pos = atom.position
                                    frac = np.dot(pos, inv_cell_contcar)
                                    atom_data_contcar.append({
                                        'Atom': f"{atom.symbol}{i}",
                                        'Element': atom.symbol,
                                        'X': f"{pos[0]:.3f}",
                                        'Y': f"{pos[1]:.3f}",
                                        'Z': f"{pos[2]:.3f}",
                                        'Frac_X': f"{frac[0]:.3f}",
                                        'Frac_Y': f"{frac[1]:.3f}",
                                        'Frac_Z': f"{frac[2]:.3f}"
                                    })

                                df_contcar_atoms = pd.DataFrame(atom_data_contcar)
                                st.dataframe(df_contcar_atoms, height=300, use_container_width=True)

                    else:
                        struct_name = list(structures.keys())[0]
                        struct = structures[struct_name]

                        ase_struct = AseAtomsAdaptor.get_atoms(struct)
                        xyz_io = StringIO()
                        write(xyz_io, ase_struct, format="xyz")
                        xyz_str = xyz_io.getvalue()
                        view = py3Dmol.view(width=600, height=500)
                        view.addModel(xyz_str, "xyz")
                        apply_atom_style(view, atom_style)
                        add_box(view, ase_struct.get_cell(), color='black', linewidth=0.05)

                        if show_cell_vectors:
                            add_cell_vectors(view, ase_struct.get_cell())

                        if show_labels:
                            for i, atom in enumerate(ase_struct):
                                view.addLabel(f"{atom.symbol}{i}", {
                                    'position': {'x': atom.position[0], 'y': atom.position[1], 'z': atom.position[2]},
                                    'backgroundColor': 'white',
                                    'fontColor': 'black',
                                    'fontSize': 10,
                                    'borderThickness': 0.5,
                                    'borderColor': 'grey'
                                })

                        view.zoomTo()
                        view.zoom(1.1)
                        html_viz = view._make_html()
                        st.components.v1.html(html_viz, height=520)

                        if show_labels:
                            st.subheader("Atomic Positions")

                            atom_data = []
                            cell = ase_struct.get_cell()
                            inv_cell = np.linalg.inv(cell) if np.linalg.det(cell) > 1e-9 else np.eye(3)

                            for i, atom in enumerate(ase_struct):
                                pos = atom.position
                                frac = np.dot(pos, inv_cell)
                                atom_data.append({
                                    'Atom': f"{atom.symbol}{i}",
                                    'Element': atom.symbol,
                                    'X (Å)': f"{pos[0]:.3f}",
                                    'Y (Å)': f"{pos[1]:.3f}",
                                    'Z (Å)': f"{pos[2]:.3f}",
                                    'Frac_X': f"{frac[0]:.3f}",
                                    'Frac_Y': f"{frac[1]:.3f}",
                                    'Frac_Z': f"{frac[2]:.3f}"
                                })

                            df_atoms = pd.DataFrame(atom_data)
                            st.dataframe(df_atoms, height=400, use_container_width=True)

                except Exception as e:
                    st.error(f"Error loading structure: {str(e)}")
                    import traceback

                    st.error(traceback.format_exc())

            tab_idx += 1


        if incar_found:
            with tabs[tab_idx]:
                st.header("INCAR Parameters Analysis")

                try:
                    params = parse_incar_detailed(incar_content)

                    if not params:
                        st.warning("No parameters found in INCAR file or file is empty")
                    else:
                        st.info(f"Found {len(params)} parameters in INCAR file")

                        categories = {}
                        for param, value in params.items():
                            explanation = get_parameter_explanation(param, value)
                            category = explanation['category']
                            if category not in categories:
                                categories[category] = []
                            categories[category].append((param, value, explanation))

                        st.subheader("Parameters by Category")

                        for category in sorted(categories.keys()):
                            with st.expander(f"**{category}** ({len(categories[category])} parameters)", expanded=True):
                                for param, value, explanation in categories[category]:
                                    st.markdown(f"### `{param} = {value}`")
                                    st.write(f"**Description:** {explanation['description']}")
                                    st.write(f"**Interpretation:** {explanation['value_meaning']}")
                                    st.divider()

                        st.subheader("Complete INCAR File")
                        st.code(incar_content, language='text')

                        st.download_button(
                            label="Download INCAR File",
                            data=incar_content,
                            file_name="INCAR",
                            mime="text/plain",
                            type='primary'
                        )

                        st.subheader("Quick Summary")

                        summary_items = []

                        if 'IBRION' in params:
                            ibrion_val = params['IBRION']
                            if ibrion_val == '-1':
                                summary_items.append("**Calculation Type:** Single-point energy calculation")
                            elif ibrion_val in ['1', '2', '3']:
                                summary_items.append("**Calculation Type:** Geometry optimization")
                            elif ibrion_val == '0':
                                summary_items.append("**Calculation Type:** Molecular dynamics")
                            else:
                                summary_items.append("**Calculation Type:** Phonon/vibrational calculation")

                        if 'ISPIN' in params:
                            if params['ISPIN'] == '2':
                                summary_items.append("**Magnetism:** Spin-polarized calculation enabled")
                            else:
                                summary_items.append("**Magnetism:** Non-magnetic calculation")

                        if 'ENCUT' in params:
                            summary_items.append(f"**Energy Cutoff:** {params['ENCUT']} eV")

                        if 'EDIFF' in params:
                            summary_items.append(f"**Electronic Convergence:** {params['EDIFF']} eV")

                        if 'PREC' in params:
                            summary_items.append(f"**Precision:** {params['PREC']}")

                        if 'ISMEAR' in params and 'SIGMA' in params:
                            summary_items.append(f"**Smearing:** Method {params['ISMEAR']}, σ = {params['SIGMA']} eV")

                        if 'ALGO' in params:
                            summary_items.append(f"**Algorithm:** {params['ALGO']}")


                        if 'IVDW' in params and params['IVDW'] != '0':
                            summary_items.append(f"**Van der Waals:** Enabled (method {params['IVDW']})")

                        for item in summary_items:
                            st.markdown(item)

                except Exception as e:
                    st.error(f"Error processing INCAR file: {str(e)}")

else:
    with st.expander("📁 File Format Information", expanded = True):
        st.markdown("""
            ### Supported Files

            **📊 OSZICAR**
            - Electronic and ionic step convergence
            - Energy evolution during relaxation
            - Electronic step counts per ionic step
            - Method usage (DAV/RMM)

            **📈 DOSCAR**
            - Total DOS data
            - Spin-polarized calculations supported
            - Fermi energy extraction

            **🔬 VASPRUN.xml**
            - DOS extraction via pymatgen
            - Electronic band structure data
            - Ionic step history
            - Energy, forces, and stress tensors
            - Lattice parameters per step
            - Initial and final structures

            **🌊 EIGENVAL**
            - Eigenvalues at each k-point
            - Band structure data
            - Occupation numbers

            **⚙️ INCAR**
            - EDIFF convergence threshold
            - All calculation parameters
            - Parameter categories and descriptions

            **🎯 KPOINTS**
            - K-point mesh definition
            - Required for band structure paths with vasprun.xml

            ---

            ### File Naming
            Files are recognized by names containing (case-insensitive):
            - 'oszicar' → OSZICAR
            - 'doscar' → DOSCAR
            - 'vasprun' → VASPRUN.xml
            - 'eigenval' → EIGENVAL
            - 'incar' → INCAR
            - 'kpoints' → KPOINTS

            ---

            ### Analysis Tabs

            **📊 OSZICAR Analysis**
            - Energy vs ionic step plots
            - Energy difference tracking
            - Electronic steps per ionic step
            - Convergence checking against EDIFF

            **📈 DOSCAR Analysis**
            - DOS plots with adjustable energy range
            - DOS at Fermi level
            - Metallic vs semiconducting classification

            **🔬 VASPRUN DOS**
            - DOS extraction using pymatgen
            - Fermi level alignment
            - CSV export

            **🎵 VASPRUN Bands**
            - Band structure plots
            - Band gap calculation (direct/indirect)
            - Requires k-point path (not Gamma-only)
            - Use with KPOINTS file

            **⚛️ VASPRUN Ionic Steps**
            - Energy per ionic step
            - Maximum force per step
            - Lattice parameters (a, b, c, α, β, γ)
            - Cell volume changes

            **🌊 Band Structure (EIGENVAL)**
            - Band structure from EIGENVAL
            - Works with any k-point mesh
            - VBM/CBM identification

            **⚙️ INCAR Parameters**
            - Parameter explanations
            - Value interpretations
            - Grouped by category

            ---

            ### Notes
            - Upload multiple files for combined analysis
            - Band structure from vasprun.xml requires KPOINTS file with k-path
            - EIGENVAL works for band structure with standard k-meshes
            - All plots can be downloaded as CSV
            """)
