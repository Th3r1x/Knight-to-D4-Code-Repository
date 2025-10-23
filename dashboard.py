import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import time
from io import StringIO

# --- Import Solver Logic ---
# This assumes solver.py is in the same directory.
try:
    from solver2 import (
        MDDVRPSRC_Environment,
        TBIH_1, TBIH_2, TBIH_3, TBIH_4, TBIH_5,
        PDSRASolver
    )
except ImportError:
    st.error("`solver.py` not found. Please ensure it's in the same directory as this dashboard.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title = "MDDVRPSRC Dashboard with Plotly",
    page_icon = "🚚",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# --- Helper Functions ---

@st.cache_data
def load_default_data():
    """Loads default sample data into pandas DataFrames."""
    nodes_csv = """id,type,lat,lon,demand
0,depot,14.5547,121.0244,0
1,depot,14.5678,121.0333,0
2,shelter,14.5550,121.0340,100
3,shelter,14.5600,121.0200,150
4,shelter,14.5700,121.0250,80
5,connecting,14.5620,121.0280,0
6,connecting,14.5650,121.0400,0
"""

    edges_csv = """u,v,weight,max_capacity,time,damage
0,5,0.9,10,5,1
1,6,0.8,10,4,0
2,5,1.0,8,3,2
2,6,0.7,8,3,3
3,5,0.9,8,6,1
4,5,1.2,8,4,0
4,6,1.0,8,5,2
5,6,1.3,10,3,1
"""

    nodes_df = pd.read_csv(StringIO(nodes_csv)).set_index('id')
    edges_df = pd.read_csv(StringIO(edges_csv))
    
    return nodes_df, edges_df

def create_graph_from_dfs(nodes_df, edges_df):
    """Creates a NetworkX graph from the node and edge DataFrames."""
    G = nx.Graph()
    for idx, row in nodes_df.iterrows():
        G.add_node(
            int(idx),
            type = row['type'],
            pos = (row['lon'], row['lat']),
            demand = int(row['demand'])
        )
    
    for _, row in edges_df.iterrows():
        G.add_edge(
            int(row['u']), int(row['v']),
            weight = float(row['weight']),
            time = float(row['time']),
            max_capacity = int(row['max_capacity']),
            damage = int(row.get('damage', 0))
        )
    
    return G

def run_simulation(graph, vehicle_configs, heuristic_func, solver_params):
    """Executes the vehicle routing simulation."""
    with st.spinner("Running simulation..."):
        env = MDDVRPSRC_Environment(graph, vehicle_configs)
        solver = PDSRASolver(heuristic_func=heuristic_func, **solver_params)
        max_steps = 1000
        for _ in range(max_steps):
            if env.all_demands_met():
                break
            
            actions = solver.decide_actions(env)
            env.step(actions)
        else:
            st.warning(f"Simulation stopped after reaching the {max_steps} of steps.")
    
    # Prepend the initial state for a clean start at step 0 of the animation
    return [env.history[0]] + env.history
def create_plotly_map(history, nodes_df, edges_df, current_time):
    """Creates a Plotly map with shelters on top (semi-transparent), vehicles beneath."""
    if not history:
        return go.Figure()

    fig = go.Figure()

    # --- Find current step based on time ---
    event_step_index = 0
    for i, step in enumerate(history):
        if step['time'] <= current_time:
            event_step_index = i
        else:
            break
    current_state = history[event_step_index]

    # --- 1️⃣ Road Network (bottom layer) ---
    edge_lons, edge_lats = [], []
    for _, edge in edges_df.iterrows():
        if edge['u'] in nodes_df.index and edge['v'] in nodes_df.index:
            u_node, v_node = nodes_df.loc[edge['u']], nodes_df.loc[edge['v']]
            edge_lons.extend([u_node['lon'], v_node['lon'], None])
            edge_lats.extend([u_node['lat'], v_node['lat'], None])

    fig.add_trace(go.Scattermap(
        lon=edge_lons, lat=edge_lats, mode='lines',
        line=dict(width=1.5, color='#999'),
        name='Road Network', hoverinfo='none'
    ))

    # --- 2️⃣ Vehicle Paths (middle layer) ---
    num_vehicles = len(history[0]['vehicles'])
    vehicle_colors = ['orange']
    for i in range(num_vehicles):
        path_lons, path_lats = [], []
        for step_idx in range(1, event_step_index + 1):
            v_state = next((v for v in history[step_idx]['vehicles'] if v['id'] == i), None)
            prev_v_state = next((v for v in history[step_idx-1]['vehicles'] if v['id'] == i), None)
            if v_state and prev_v_state and prev_v_state['location'] != v_state['location']:
                u, v = prev_v_state['location'], v_state['location']
                if u in nodes_df.index and v in nodes_df.index:
                    u_node, v_node = nodes_df.loc[u], nodes_df.loc[v]
                    path_lons.extend([u_node['lon'], v_node['lon'], None])
                    path_lats.extend([u_node['lat'], v_node['lat'], None])
        fig.add_trace(go.Scattermap(
            lon=path_lons, lat=path_lats, mode='lines',
            line=dict(width=3, color=vehicle_colors[i % len(vehicle_colors)]),
            name=f'Vehicle {i} Path', hoverinfo='name'
        ))

    # --- 3️⃣ Vehicle Current Positions (middle layer) ---
    current_event_time = current_state['time']
    vehicle_plot_data = []
    for v_state in current_state['vehicles']:
        loc_id = v_state['location']
        if v_state['is_moving'] and v_state['arrival_time'] > current_event_time:
            start_id, end_id = v_state['location'], v_state['destination']
            if start_id in nodes_df.index and end_id in nodes_df.index:
                start_pos, end_pos = nodes_df.loc[start_id], nodes_df.loc[end_id]
                duration = v_state['arrival_time'] - current_event_time
                progress = (current_time - current_event_time) / duration if duration > 0 else 1.0
                lon = start_pos['lon'] + (end_pos['lon'] - start_pos['lon']) * progress
                lat = start_pos['lat'] + (end_pos['lat'] - start_pos['lat']) * progress
            else:
                pos = nodes_df.loc[loc_id]; lon, lat = pos['lon'], pos['lat']
        else:
            pos = nodes_df.loc[loc_id]; lon, lat = pos['lon'], pos['lat']

        vehicle_plot_data.append({
            'id': v_state['id'],
            'lon': lon, 'lat': lat,
            'tooltip': (
                f"<b>Vehicle {v_state['id']}</b><br>"
                f"Cap: {v_state.get('capacity', 'N/A')}/{v_state.get('max_capacity', 'N/A')}<br>"
                f"Dest: {v_state.get('destination', 'N/A')}"
            ),
            'color': vehicle_colors[v_state['id'] % len(vehicle_colors)]
        })

    if vehicle_plot_data:
        plot_df = pd.DataFrame(vehicle_plot_data)
        fig.add_trace(go.Scattermap(
            lon=plot_df['lon'], lat=plot_df['lat'], mode='markers',
            marker=dict(size=18, color=plot_df['color'], symbol='circle'),
            text=plot_df['tooltip'], hoverinfo='text', name='Vehicles'
        ))

    # --- 4️⃣ Shelters and Depots (TOP LAYER) ---
    current_demands = current_state.get('demands', {})
    node_colors, node_opacity = [], []

    for node_id in nodes_df.index:
        node_type = nodes_df.loc[node_id, 'type']
        if node_type == 'shelter':
            remaining = current_demands.get(node_id, nodes_df.loc[node_id, 'demand'])
            color = '#28a745' if remaining <= 0 else '#dc3545'
            opacity = 0.75  # semi-transparent on top
        elif node_type == 'depot':
            color, opacity = '#007bff', 0.9
        else:
            color, opacity = '#6c757d', 0.9
        node_colors.append(color)
        node_opacity.append(opacity)

    fig.add_trace(go.Scattermap(
        lon=nodes_df['lon'],
        lat=nodes_df['lat'],
        mode='markers',
        marker=dict(size=13, color=node_colors, opacity=node_opacity),
        text=[
            f"<b>ID:</b> {idx}<br><b>Type:</b> {row['type']}<br><b>Demand:</b> {current_demands.get(idx, row['demand'])}"
            for idx, row in nodes_df.iterrows()
        ],
        hoverinfo='text',
        name='Shelters / Depots'
    ))

    # --- Layout ---
    fig.update_layout(
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        map_style="carto-positron",
        map_zoom=13,
        map_center={"lat": nodes_df["lat"].mean(), "lon": nodes_df["lon"].mean()},
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    return fig




# --- App State Management ---
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = None
if 'current_event_step' not in st.session_state:
    st.session_state.current_event_step = 0
if 'animation_time' not in st.session_state:
    st.session_state.animation_time = 0.0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'nodes_df' not in st.session_state:
    st.session_state.nodes_df, st.session_state.edges_df = load_default_data()

# --- UI Layout ---
with st.sidebar:
    st.header("1. Data Input")
    input_method = st.radio("Choose input method", ["Use Sample Data", "Upload CSV", "Manual Input"], key="input_method")
    nodes_df_init, edges_df_init = load_default_data()
    if input_method == "Upload CSV":
        uploaded_nodes, uploaded_edges = st.file_uploader("Upload Nodes CSV", type="csv"), st.file_uploader("Upload Edges CSV", type="csv")
        if uploaded_nodes and uploaded_edges:
            st.session_state.nodes_df, st.session_state.edges_df = pd.read_csv(uploaded_nodes).set_index('id'), pd.read_csv(uploaded_edges)
    elif input_method == "Manual Input":
        st.subheader("Node Data"), 
        st.session_state.nodes_df = st.data_editor(st.session_state.nodes_df.reset_index(), num_rows="dynamic").set_index('id')
        st.subheader("Edge Data")
        st.session_state.edges_df = st.data_editor(st.session_state.edges_df, num_rows="dynamic")
    else:
        st.session_state.nodes_df, st.session_state.edges_df = nodes_df_init, edges_df_init
    st.divider()
    st.header("2. Vehicle Configuration")
    num_vehicles = st.number_input("Number of Vehicles", 1, 50, 2)
    vehicle_configs, all_node_ids = [], list(st.session_state.nodes_df.index)
    if all_node_ids:
        for i in range(num_vehicles):
            st.subheader(f"Vehicle {i+1}"); cols = st.columns(2)
            capacity = cols[0].number_input(f"Capacity##{i}", 1, 1000, 100)
            start_loc = cols[1].selectbox(f"Start Location##{i}", all_node_ids, index=i % len(all_node_ids))
            vehicle_configs.append((capacity, start_loc))
    st.divider()
    st.header("3. Solver Settings")
    heuristic_map = { "TBIH-1 (Random)": TBIH_1, "TBIH-2 (DSIH)": TBIH_2, "TBIH-3 (DCW)": TBIH_3,
                      "TBIH-4 (DLA-SIH)": TBIH_4, "TBIH-5 (DLA-CW)": TBIH_5 }
    heuristic_func = heuristic_map[st.selectbox("Select Heuristic", list(heuristic_map.keys()))]
    solver_params = {'num_simulations': st.slider("PDS-RA Simulations", 1, 10, 3),
                     'lookahead_horizon': st.slider("PDS-RA Lookahead Horizon", 1, 20, 7)}
    st.divider()
    if st.button("▶️ Run Simulation", type="primary", use_container_width=True, disabled=not all_node_ids):
        graph = create_graph_from_dfs(st.session_state.nodes_df, st.session_state.edges_df)
        st.session_state.simulation_history = run_simulation(graph, vehicle_configs, heuristic_func, solver_params)
        st.session_state.simulation_run, st.session_state.current_event_step = True, 0
        st.session_state.animation_time, st.session_state.is_playing = 0.0, False
        st.rerun()

# --- Main Panel ---
st.title("AGOS: Aid & Goods Optimization System")
if not st.session_state.simulation_run:
    st.info("Configure the simulation in the sidebar and click 'Run Simulation'.")
    st.image("https://placehold.co/800x400/FFFFFF/262730?text=Results+Appear+Here")
else:
    history = st.session_state.simulation_history
    max_steps = len(history) - 1 if history else 0
    max_time = history[-1]['time'] if history else 0.0

    # --- State Update Logic (from buttons) ---
    # This block runs before widgets are drawn, preventing the API exception.
    if 'step_action' in st.session_state:
        action = st.session_state.pop('step_action')
        if action == 'first':
            st.session_state.current_event_step = 0
        elif action == 'prev':
            st.session_state.current_event_step = max(0, st.session_state.current_event_step - 1)
        elif action == 'next':
            st.session_state.current_event_step = min(max_steps, st.session_state.current_event_step + 1)
        
        # Sync animation time after a step change
        st.session_state.is_playing = False
        if st.session_state.simulation_history:
            st.session_state.animation_time = st.session_state.simulation_history[st.session_state.current_event_step]['time']

    st.header("Simulation Summary"); cols = st.columns(4)
    if max_steps > 0:
        cols[0].metric("Total Distance (km)", f"{history[-1]['total_distance']:.2f}")
        cols[1].metric("Total Time (min)", f"{history[-1]['time']:.2f}")
    cols[2].metric("Total Events", max_steps); cols[3].metric("Vehicles", len(history[0]['vehicles']))
    st.header("Route Visualization")
    
    def on_slider_change():
        # This callback handles manual slider drags by the user
        st.session_state.is_playing = False
        if st.session_state.simulation_history:
            st.session_state.animation_time = st.session_state.simulation_history[st.session_state.current_event_step]['time']

    st.slider("Simulation Event", 0, max_steps, key='current_event_step', on_change=on_slider_change)
    
    cols = st.columns([2, 1, 1, 1, 1, 2])
    if cols[1].button("⏮️", use_container_width=True):
        st.session_state.step_action = 'first'
        st.rerun()
    if cols[2].button("⏪", use_container_width=True):
        st.session_state.step_action = 'prev'
        st.rerun()
    if cols[3].button("⏯️", use_container_width=True): 
        st.session_state.is_playing = not st.session_state.is_playing
        # Sync time to current step when play is pressed
        if st.session_state.is_playing:
             st.session_state.animation_time = history[st.session_state.current_event_step]['time']
        st.rerun()
    if cols[4].button("⏩", use_container_width=True):
        st.session_state.step_action = 'next'
        st.rerun()

    try:
        fig = create_plotly_map(history, st.session_state.nodes_df, st.session_state.edges_df, st.session_state.animation_time)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate map: {e}"); st.exception(e)

    with st.expander("Show Detailed Route History"):
        if max_steps > 0:
            for i, step in enumerate(history[1:], 1):
                st.markdown(f"**Step {i} (Time: {step['time']:.2f})**")
                for v in step['vehicles']:
                    st.markdown(f"  - **Vehicle {v['id']}**: At Node `{v['location']}` → `{v['destination']}`. Capacity: {v['capacity']}/{v['max_capacity']}. Status: {'Moving' if v['is_moving'] else 'Idle'}.")
    
    if st.session_state.is_playing:
        if st.session_state.animation_time < max_time:
            time_increment = max_time / 200 if max_time > 0 else 1.0
            st.session_state.animation_time = min(max_time, st.session_state.animation_time + time_increment)
            new_event_step = next((i for i, step in reversed(list(enumerate(history))) if step['time'] <= st.session_state.animation_time), 0)
            
            # Update slider only if the step actually changes
            if st.session_state.current_event_step != new_event_step:
                st.session_state.current_event_step = new_event_step

            time.sleep(0.04) 
            st.rerun()
        else:
            st.session_state.is_playing = False; st.rerun()