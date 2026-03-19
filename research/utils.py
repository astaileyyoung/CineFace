import logging
import itertools

import cv2
import numpy as np
import pandas as pd 
import sqlalchemy as db
import plotly.express as px
import plotly.graph_objs as go
import scipy.ndimage as ndimage
from scipy.spatial import distance


handler = logging.StreamHandler()
handler.setLevel(20)
formatter = logging.Formatter('[%(asctime)s] [utils] [%(levelname)s]: %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger = logging.getLogger("utils")
logger.addHandler(handler)
logger.setLevel(20)


layout = {
    'title': {
        'text': '',
        'font': {'size': 22, 'family': 'Raleway', 'color': 'white'},
        'x': 0.5,
        'y': 0.9,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    'xaxis': {
        'title': {
            'text': '',
            'font': {'size': 18, 'family': 'Raleway', 'color': 'white'} 
        },
        'tickfont': {'size': 14, 'family': 'Roboto', 'color': 'white'}
    },
    'yaxis': {
        'layer': 'below traces',
        'title': {
            'text': '',
            'font': {'size': 18, 'family': 'Raleway', 'color': 'white'} 
        },
        'tickfont': {'size': 14, 'family': 'Roboto', 'color': 'white'}
    },
    'font': {'color': 'white'},
    'paper_bgcolor': '#787577',
    'plot_bgcolor': 'rgba(61, 61, 61, 0)'
}


def create_gridmap_from_director(name, conn, layout=None, size=600):
    query = f"""
        SELECT * 
        FROM CineFaceDW.vwGridByJob gbj
        WHERE gbj.name = '{name}' AND gbj.job_name = 'Director'
        """
    df = pd.read_sql_query(query, conn)
    temp = df[[x for x in df.columns if "pct_faces" in x]]
    temp = temp.mean()
    grid = temp.values.reshape(3, 3)

    fig = px.imshow(np.round(grid, 3),
                    x=['Left', 'Middle', 'Right'],
                    y=['Top', 'Center', 'Bottom'],
                    labels={'color': 'Face Percent by Section'},
                    text_auto=True,
                    aspect='auto',
                    color_continuous_scale='Aggrnyl')
    if layout:
        fig.update_layout(layout)
    fig.update_layout(title={'text': f'Gridmap of Face Locations in {name}',
                            'y': 0.95},
                      width=size,
                      height=size,
                      xaxis=dict(side="bottom"),
                      yaxis=dict(autorange="reversed"), 
                      margin=dict(l=50, r=50, t=80, b=50),
                      coloraxis_colorbar=dict(title="Delta %"),    
                    )
    fig.show()


def get_sample_grid(engine):
    with engine.connect() as conn:
        query = """
            SELECT 
                AVG(pct_tl), AVG(pct_tc) as tc, AVG(pct_tr) as tr,
                AVG(pct_ml), AVG(pct_mc) as mc, AVG(pct_mr) as mr,
                AVG(pct_bl), AVG(pct_bc) as bc, AVG(pct_br) as br
            FROM factWork
        """
        df = pd.read_sql_query(db.text(query), conn)
    grid = df.values.reshape(3, 3)
    grid_norm = grid / grid.sum()
    return (grid_norm * 100).round(1)


def get_director_grid(name, engine):
    with engine.connect() as conn:
        query = f"""
            SELECT 
                AVG(pct_tl), AVG(pct_tc) as tc, AVG(pct_tr) as tr,
                AVG(pct_ml), AVG(pct_mc) as mc, AVG(pct_mr) as mr,
                AVG(pct_bl), AVG(pct_bc) as bc, AVG(pct_br) as br
            FROM vwWorksByDirector
            WHERE person_name = '{name}'
        """
        df = pd.read_sql_query(db.text(query), conn)
    grid = df.values.reshape(3, 3)
    grid_norm = grid / grid.sum()
    return (grid_norm * 100).round(1)


def plot_grid(grid, plot_title=None, width=600, height=600, layout=None, dst=None, transparent=False):
    # Center the color scale at zero
    limit = max(abs(grid.min()), abs(grid.max()))
    
    fig = px.imshow(
        grid,
        x = ['Left', 'Middle', 'Right'],
        y = ['Top', 'Center', 'Bottom'],
        color_continuous_scale='Aggrnyl', 
        # range_color=[-limit, limit],
        text_auto=".1f", # This shows the difference value in each box
        title=plot_title if plot_title else "",
        aspect="equal"
    )

    if layout:
        fig.update_layout(layout)

    unified_layout = {
        "title": {
            "text": plot_title if plot_title else "",
            "x": 0.5, "xanchor": "center", "yanchor": "top"
        },
        "width": width,
        "height": height,
        "xaxis": dict(side="bottom", title="Horizontal Position"),
        "yaxis": dict(autorange="reversed", title="Vertical Position"), 
        "margin": dict(l=50, r=50, t=100, b=50), # Increased top margin for Canva title safety
        "coloraxis_colorbar": dict(title="%")
    }

    fig.update_layout(unified_layout)
    
    fig.show()
    if dst:
        if transparent:
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', # Transparent outer background
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent inner plot area
                font=dict(color="white")       # Good if your Canva theme is dark
            )
        fig.write_image(dst, width=width, height=height, scale=2)


def plot_sample_grid(engine, plot_title=None, layout=None, dst=None):
    grid_norm = get_sample_grid(engine)
    plot_grid(grid_norm, plot_title=plot_title, layout=layout, dst=dst)


def plot_director_grid(engine, plot_title=None, layout=None, dst=None):
    grid_norm = get_director_grid(engine)
    plot_grid(grid_norm, plot_title=plot_title, layout=layout, dst=dst)


def compare_director_to_sample_grid(name, 
                                    engine, 
                                    title=None, 
                                    layout=None,
                                    dst=None,
                                    transparent=False,
                                    width=600,
                                    height=600):
    with engine.connect() as conn:
        query = f"""
            SELECT 
                AVG(pct_tl), AVG(pct_tc) as tc, AVG(pct_tr) as tr,
                AVG(pct_ml), AVG(pct_mc) as mc, AVG(pct_mr) as mr,
                AVG(pct_bl), AVG(pct_bc) as bc, AVG(pct_br) as br
            FROM vwGridByJob gbj
            WHERE gbj.name = '{name}' AND gbj.job_name = 'Director'
            GROUP BY gbj.name
            """
        df = pd.read_sql_query(db.text(query), conn)

    temp = df.mean()
    grid = temp.values.reshape(3, 3) * 100
    grid_norm = get_sample_grid(engine)
    diff_grid = (grid - grid_norm)
    plot_grid(diff_grid, plot_title=title, layout=layout, dst=dst, width=width, height=height, transparent=transparent)


def plot_directors_against_sample(g, 
                                  x, 
                                  y, 
                                  names, 
                                  hover_data=["name"], 
                                  name_field="name", 
                                  dst=None,
                                  width=1200,
                                  height=800,
                                  title=None,
                                  x_title=None,
                                  y_title=None,
                                  transparent=False,
                                  margin=dict(t=100, l=50, r=50, b=50)):
    colors = px.colors.qualitative.Bold
    fig = px.scatter(g, 
                    x=x,
                    y=y,
                    hover_data=hover_data,
                    color_discrete_sequence=["#79b8b8"],
                    trendline="ols")
    fig.update_layout(layout)
    fig.update_layout(
        xaxis=dict(
            title=x_title if x_title else x), 
        yaxis=dict(
            title=y_title if y_title else y), 
        title=dict(text=title),
        margin=margin
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    fig.update_traces(selector=dict(mode="lines"), line=dict(dash="dash", color="#d81275", width=5), name="Trend")
    for num, name in enumerate(names):
        temp = g[g[name_field] == name]
        fig.add_trace(go.Scatter(x=temp[x], 
                                y=temp[y], 
                                marker=dict(
                                    color=colors[num], 
                                    size=16, 
                                    line=dict(
                                        color="white",
                                        width=2
                                    )
                                    ), name=name))
    fig.show()
    
    if dst:
        if transparent:
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', # Transparent outer background
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent inner plot area
                font=dict(color="white")       # Good if your Canva theme is dark
            )
        fig.write_image(dst, width=width, height=height, scale=2)


def plot_directors_titles_sample(df, 
                                 x, 
                                 y, 
                                 name, 
                                 layout=None, 
                                 title=None,
                                 x_title=None,
                                 y_title=None,
                                 line=False, 
                                 titles=None, 
                                 trendline=False,
                                 trendline_options=None,
                                 dst=None, 
                                 transparent=False, 
                                 width=800, 
                                 height=600):
    temp = df[df['person_name'] == name]
    if titles:
        temp = temp[temp['title'].isin(titles)]
    sample = df.drop(temp.index)
    if not line:
        if trendline:
            if trendline_options:
                fig = px.scatter(sample, x=x, y=y,
                        opacity=0.3,
                        color_discrete_sequence=["white"],
                        hover_name="title",
                        trendline=trendline,
                        trendline_options=trendline_options)
            else:
                fig = px.scatter(sample, x=x, y=y,
                        opacity=0.3,
                        color_discrete_sequence=["white"],
                        hover_name="title",
                        trendline=trendline)
                
            fig.update_traces(selector=dict(mode="lines"), line=dict(dash="dash", color="#d81275", width=5), name="Trend")
        else:
            fig = px.scatter(sample, x=x, y=y,
                    opacity=0.3,
                    color_discrete_sequence=["white"],
                    hover_name="title")
        fig.add_trace(
            px.scatter(
                temp, 
                x=x,
                y=y,
                hover_name="title"
            ).update_traces(
                marker=dict(size=12, color='red', symbol='circle', line=dict(width=2, color='DarkSlateGrey')),
                name=name # Label for the legend
            ).data[0]
        )
    else:
        fig = px.line(sample, x=x, y=y,
                color_discrete_sequence=["white"],
                hover_name="title")
        fig.add_trace(
            px.line(
                temp, 
                x=x,
                y=y,
                hover_name="title"
            ).update_traces(
                marker=dict(size=12, color='red', symbol='circle', line=dict(width=2, color='DarkSlateGrey')),
                name=name # Label for the legend
            ).data[0]
        )
    
    if layout:
        fig.update_layout(layout)
    
    fig.update_layout(xaxis=dict(title=x if not x_title else x_title), 
                      yaxis=dict(title=y if not y_title else y_title),
                      title=dict(text=title))
    
    if dst:
        if transparent:
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', # Transparent outer background
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent inner plot area
                font=dict(color="white")       # Good if your Canva theme is dark
            )
        fig.write_image(dst, width=width, height=height, scale=2)
    fig.show()


def plot_director_films(df, 
                        x, 
                        y, 
                        name, 
                        hover_data=["title"], 
                        marker_labels=False,
                        transparent=None, 
                        line=False, 
                        title=None, 
                        x_name=None, 
                        y_name=None, 
                        width=800,
                        height=600,
                        text_size=14,
                        dst=None,
                        trace_names=None):
    # Ensure y is a list
    y_list = [y] if not isinstance(y, list) else y
    x_col = x[0] if isinstance(x, list) else x

    fig = go.Figure()
    temp = df[df['person_name'] == name].copy()
    temp = temp.sort_values(by=x_col)

    # x_spread = temp[x_col].max() - temp[x_col].min()
    # y_spread = temp[y_val].max() - temp[y_val].min()

    # Manually add a trace for every Y variable
    for n, y_val in enumerate(y_list):
        if line:
            # Use go.Scatter for direct control so traces don't overwrite
            fig.add_trace(go.Scatter(
                x=temp[x_col],
                y=temp[y_val],
                name=y_val if not trace_names else trace_names[y_val],  # This creates the legend entry
                mode='lines+markers' if not marker_labels else 'lines+markers+text',
                hovertext=temp['title'],
                text=temp['title'] if marker_labels and n == 0 else None,
                textposition="top right",
                hovertemplate="<b>%{hovertext}</b><br>Value: %{y}<extra></extra>",
                textfont=dict(size=text_size, color="white", shadow="auto"))
            )
        else:
            fig.add_trace(go.Scatter(
                x=temp[x_col],
                y=temp[y_val],
                name=y_val if not trace_names else trace_names[y_val],
                mode='markers',
                hovertext=temp['title']
            ))

    # Apply your specific styling
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")), line=dict(width=6), textposition="top right", cliponaxis=False)
    
    # Use your specific pink for the first line, teal for the second
    if len(fig.data) > 0: fig.data[0].line.color = "#d81275"
    if len(fig.data) > 1: fig.data[1].line.color = "#79b8b8"

    fig.update_layout(layout)
    fig.update_layout(
        xaxis=dict(title=x if not x_name else x_name), 
        yaxis=dict(title=y if not y_name else y_name), 
        title=dict(
            text=f"{x} vs. {y}" if not title else title,
            y=0.95
        )
    )
    
    fig.show()
    if dst:
        if transparent:
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', # Transparent outer background
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent inner plot area
                font=dict(color="white")       # Good if your Canva theme is dark
            )
        fig.write_image(dst, width=width, height=height, scale=2)
    

def plot_directors_3d(df, director_names, features=None):
    """
    Plots a 3D scatter of all directors as a background cloud and 
    highlights specific directors as individual traces.
    """
    if features is None:
        features = {
            'x': 'z_size_g_mean', 
            'y': 'z_v_dist_g_mean', 
            'z': 'z_gini_g_mean'
        }
        
    fig = go.Figure()

    # 1. Background "Cloud" (All other directors)
    df_others = df[~df['name'].isin(director_names)]
    fig.add_trace(go.Scatter3d(
        x=df_others[features['x']],
        y=df_others[features['y']],
        z=df_others[features['z']],
        mode='markers',
        name='Others',
        hovertext=df_others['name'],
        marker=dict(size=2, color='rgba(200, 200, 200, 0.3)', opacity=0.4)
    ))

    # 2. Cycle through colors and symbols for the highlights
    colors = px.colors.qualitative.Plotly  # or ['cyan', 'magenta', 'yellow', 'lime']
    symbols = ['diamond', 'circle', 'square', 'cross', 'hexagon']
    
    # Use zip/cycle to handle more names than colors/symbols
    for name, color, symbol in zip(director_names, itertools.cycle(colors), itertools.cycle(symbols)):
        d_data = df[df['name'] == name]
        
        if not d_data.empty:
            fig.add_trace(go.Scatter3d(
                x=d_data[features['x']],
                y=d_data[features['y']],
                z=d_data[features['z']],
                mode='markers+text',
                name=name,
                text=[name],
                textposition="top center",
                marker=dict(
                    size=10, 
                    color=color, 
                    symbol=symbol,
                    line=dict(color='white', width=2)
                )
            ))
        else:
            print(f"Warning: {name} not found in dataframe.")

    # 3. Layout Styling
    fig.update_layout(
        template="plotly_dark",
        title=f"3D Comparison: {' vs '.join(director_names)}",
        scene=dict(
            xaxis_title=features['x'],
            yaxis_title=features['y'],
            zaxis_title=features['z']
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig


def imdb_from_title(title, engine):
    query = """
        SELECT 
            imdb_id
        FROM dimWork
        WHERE title = :title
    """
    with engine.connect() as conn:
        imdb_id = conn.execute(db.text(query), {"title": title}).fetchone()[0]
    return imdb_id


def calculate_peak_density(face_coords, radius=0.05):
    """
    face_coords: Nx2 array of normalized (x, y) centers (0.0 to 1.0)
    radius: The circular 'buffer' around the peak (0.05 = 5% of screen)
    """
    # 1. Create a 2D Histogram to find the 'Max Pixel Point' area
    # We use a 50x50 grid to find the general 'Peak'
    heatmap, xedges, yedges = np.histogram2d(
        face_coords[:, 0], face_coords[:, 1], bins=50, range=[[0, 1], [0, 1]]
    )
    
    # 2. Find the coordinates of the Peak Bin
    max_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    peak_x = (xedges[max_idx[0]] + xedges[max_idx[0] + 1]) / 2
    peak_y = (yedges[max_idx[1]] + yedges[max_idx[1] + 1]) / 2
    peak_point = np.array([peak_x, peak_y])

    # 3. Calculate distance of ALL faces from this Peak Point
    distances = distance.cdist(face_coords, [peak_point], 'euclidean').flatten()

    # 4. Calculate Density Metrics
    total_faces = len(face_coords)
    faces_in_peak = np.sum(distances <= radius)
    
    peak_concentration = (faces_in_peak / total_faces) * 100
    avg_dist_from_peak = np.mean(distances)

    return {
        "peak_location": (peak_x, peak_y),
        "peak_concentration_pct": peak_concentration,
        "avg_dist_from_peak": avg_dist_from_peak,
        "max_bin_frequency": np.max(heatmap)
    }


def find_most_common_face_spot(df, sigmaX=30):
    h, w = int(df['img_height'].iloc[0]), int(df['img_width'].iloc[0])
    
    # Use the SAME technique as your bounding box approach!
    # But with small boxes around centers
    diff = np.zeros((h + 1, w + 1), dtype=np.float32)
    
    # Get centers
    cx = ((df['x1'] + df['x2']) / 2).astype(int).values
    cy = ((df['y1'] + df['y2']) / 2).astype(int).values
    
    # Create small boxes around each center (instead of points)
    splat_size = 10  # Half-width of splat box
    x1 = np.clip(cx - splat_size, 0, w)
    y1 = np.clip(cy - splat_size, 0, h)
    x2 = np.clip(cx + splat_size, 0, w)
    y2 = np.clip(cy + splat_size, 0, h)
    
    # Use your difference array technique
    np.add.at(diff, (y1, x1), 1)
    np.add.at(diff, (y1, x2), -1)
    np.add.at(diff, (y2, x1), -1)
    np.add.at(diff, (y2, x2), 1)
    
    # Compute cumulative sum
    heatmap = diff.cumsum(axis=0).cumsum(axis=1)[:-1, :-1]
    
    # Blur to smooth
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigmaX)
    
    # Normalize
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Find peak
    _, _, _, max_loc = cv2.minMaxLoc(heatmap)
    
    return heatmap


def draw_gridlines(img):
    h, w = img.shape[:2]
    # Use a semi-transparent or soft white/grey
    # OpenCV uses BGR: (Blue, Green, Red)
    line_color = (200, 200, 200) 
    thickness = 2

    # Vertical lines (at 1/3 and 2/3 of width)
    cv2.line(img, (int(w / 3), 0), (int(w / 3), h), line_color, thickness)
    cv2.line(img, (int(2 * w / 3), 0), (int(2 * w / 3), h), line_color, thickness)

    # Horizontal lines (at 1/3 and 2/3 of height)
    cv2.line(img, (0, int(h / 3)), (w, int(h / 3)), line_color, thickness)
    cv2.line(img, (0, int(2 * h / 3)), (w, int(2 * h / 3)), line_color, thickness)
    
    return img


def to_color(frame,
             colormap=cv2.COLORMAP_VIRIDIS):
    return cv2.applyColorMap(frame.astype(np.uint8), colormap)


def create_heatmap_from_title(title, engine, year=None, resize=False, grid=False, sigmaX=30):
    with engine.connect() as conn:
        query = f"""
            SELECT
                ff.work_id,
                ff.imdb_id,
                ff.frame_num,
                ff.x1,
                ff.y1,
                ff.x2,
                ff.y2,
                di.img_width AS img_width,
                di.img_height AS img_height,
                dw.year
            FROM factFace ff
            INNER JOIN dimWork dw ON dw.work_id = ff.work_id
            INNER JOIN dimDetectionInfo di ON ff.work_id = di.work_id
            WHERE dw.title LIKE '{title}'
        """
        if year:
            query += f" AND dw.year = {year}"
        df = pd.read_sql_query(db.text(query), conn)
    year = df.at[0, 'year']
    
    frame = find_most_common_face_spot(df, sigmaX=sigmaX)

    h, w = frame.shape[:2]
    ratio = w / h
    if resize and isinstance(resize, int):
        print("Ratio: ", ratio)
        if h > resize:
            hh = resize
            ww = int(ratio * hh)
            frame = cv2.resize(frame, (ww, hh))
    elif resize:
        print("Resize argument must be int.")
        return 
    # x, y = find_max_new(frame)
    color = to_color(frame, colormap=cv2.COLORMAP_TURBO)
    if grid:
        color = draw_gridlines(color)
        
    return color


def process_faces(df):
    h = int(df.at[0, 'img_height'])
    w = int(df.at[0, 'img_width'])
    mask = np.zeros(shape=(h, w), dtype=int)
    
    # Calculate all centers at once
    cx = ((df['x1'] + df['x2']) / 2).astype(int).values
    cy = ((df['y1'] + df['y2']) / 2).astype(int).values
    
    # Filter valid coordinates
    valid = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
    cx = cx[valid]
    cy = cy[valid]
    
    # Accumulate at center points (just increment by 1 each time)
    np.add.at(mask, (cy, cx), 1)
    
    return mask


def create_grid(mask):
    grid = np.zeros(shape=(3, 3))
    rx = int(mask.shape[1]/3)
    ry = int(mask.shape[0]/3)
    a = [x * rx for x in range(1, 4)]
    b = [x * ry for x in range(1, 4)]
    for ii, i in enumerate(a):
        for ij, j in enumerate(b):
            xi = i - rx
            yi = j - ry
            temp = mask[yi:j, xi:i].sum()
            grid[ij, ii] = temp/mask.sum()
    return grid


def create_gridmap_from_title(title, engine, year=None, layout=None, size=600):
    with engine.connect() as conn:
        query = f"""
            SELECT
                ff.imdb_id,
                ff.frame_num,
                ff.x1,
                ff.y1,
                ff.x2,
                ff.y2,
                di.img_width,
                di.img_height,
                dw.year
            FROM factFace ff
            INNER JOIN dimWork dw ON dw.work_id = ff.work_id
            INNER JOIN dimDetectionInfo di ON dw.work_id = di.work_id
            WHERE dw.title LIKE '{title}'
        """
        if year:
            query += f" AND dw.year = {year}"
            
        df = pd.read_sql_query(db.text(query), conn)
    year = df.at[0, 'year']
    
    mask = process_faces(df)
    grid = create_grid(mask)

    flat_grid = grid.flatten()
    concentration_score = np.sqrt(np.mean((flat_grid - 0.1111)**2))

    print(f"Concentration Score: {concentration_score:.4f}")

    fig = px.imshow(np.round(grid, 3),
                    x=['Left', 'Middle', 'Right'],
                    y=['Top', 'Center', 'Bottom'],
                    labels={'color': 'Face Percent by Section'},
                    text_auto=True,
                    aspect='equal',
                    color_continuous_scale='Aggrnyl'
    )
    fig.update_layout(layout)
    fig.update_layout(
        title={
            "text": title if title else "",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        width=size,
        height=size,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"), 
        margin=dict(l=50, r=50, t=80, b=50),
        coloraxis_colorbar=dict(title="Delta %"),
        xaxis_title="Horizontal Position",
        yaxis_title="Vertical Position",
        template="plotly_dark"
    )
    
    fig.show()


# def create_gridmap_from_director(director, engine, layout=None, size=600):
#     with engine.connect() as conn:
#         query = "SELECT * FROM vwFacesByDirector WHERE name = :director"
#         df = pd.read_sql_query(db.text(query), conn, params={"director": director})

#     if df.empty:
#         print(f"Could not find {director} in database.")
    
#     mask = process_faces(df)
#     grid = create_grid(mask)
#     fig = px.imshow(np.round(grid, 3),
#                     x=['Left', 'Center', 'Right'],
#                     y=['Top', 'Middle', 'Bottom'],
#                     labels={'color': 'Face Percent by Section'},
#                     text_auto=True,
#                     aspect='auto',
#                     color_continuous_scale='Aggrnyl')
#     if layout:
#         fig.update_layout(layout)
#     fig.update_layout(
#         title={
#             "text": f'Gridmap of Face Locations in films of {director})',
#             "x": 0.5,
#             "xanchor": "center",
#             "yanchor": "top"
#         },
#         width=size,
#         height=size,
#         xaxis=dict(side="bottom"),
#         yaxis=dict(autorange="reversed"), 
#         margin=dict(l=50, r=50, t=80, b=50),
#         coloraxis_colorbar=dict(title="Delta %"),
#         xaxis_title="Horizontal Position",
#         yaxis_title="Vertical Position",
#         template="plotly_dark"
#     )
#     fig.show()


def create_gridmap_from_director(director, 
                                 engine, 
                                 layout=None, 
                                 width=600,
                                 height=600,
                                 transparent=False,
                                 dst=None):
    with engine.connect() as conn:
        query = "SELECT * FROM vwFacesByDirector WHERE name = :director"
        df = pd.read_sql_query(db.text(query), conn, params={"director": director})

    if df.empty:
        print(f"Could not find {director} in database.")
        return

    # --- INTERNAL LOGIC: Normalized Face Processing ---
    # We use a 1000x1000 internal grid to normalize all aspect ratios
    res = 1000
    norm_mask = np.zeros(shape=(res, res), dtype=int)
    
    # Calculate normalized centers (0.0 to 1.0) and map to our grid
    cx = (((df['x1'] + df['x2']) / 2) / df['img_width'] * (res - 1)).astype(int).values
    cy = (((df['y1'] + df['y2']) / 2) / df['img_height'] * (res - 1)).astype(int).values
    
    # Filter valid coordinates inside our 1000x1000 space
    valid = (cx >= 0) & (cx < res) & (cy >= 0) & (cy < res)
    np.add.at(norm_mask, (cy[valid], cx[valid]), 1)
    
    # --- INTERNAL LOGIC: Create 3x3 Grid ---
    # Split the 1000x1000 mask into 9 sectors
    h_chunk = res // 3
    w_chunk = res // 3
    grid = np.zeros((3, 3))
    
    total_faces = norm_mask.sum()
    if total_faces > 0:
        for r in range(3):
            for c in range(3):
                sector = norm_mask[r*h_chunk:(r+1)*h_chunk, c*w_chunk:(c+1)*w_chunk]
                grid[r, c] = (sector.sum() / total_faces) * 100
    
    # --- PLOTLY RENDERING ---
    fig = px.imshow(np.round(grid, 3),
                    x=['Left', 'Middle', 'Right'],
                    y=['Top', 'Center', 'Bottom'],
                    labels={'color': '% of Total Faces'},
                    text_auto=".1f",
                    aspect='auto',
                    color_continuous_scale='Aggrnyl')
    
    if layout:
        fig.update_layout(layout)
        
    fig.update_layout(
        title={
            "text": f'Normalized Face Locations: {director}',
            "x": 0.5, "xanchor": "center", "yanchor": "top"
        },
        width=width, height=height,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"), 
        margin=dict(l=50, r=50, t=100, b=50),
        coloraxis_colorbar=dict(title="%"),
        xaxis_title="Horizontal Position",
        yaxis_title="Vertical Position",
        template="plotly_dark"
    )
    fig.show()
    if dst:
        if transparent:
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', # Transparent outer background
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent inner plot area
                font=dict(color="white")       # Good if your Canva theme is dark
            )
        fig.write_image(dst, width=width, height=height, scale=2)


def plot_titles(df, x, y, trendline_options=None):
    if not trendline_options:
        trendline_options = {}

    fig = px.scatter(
        df, 
        x=x, 
        y=y, 
        color_discrete_sequence=["#79b8b8"], 
        hover_data=["title", "imdb_id", "directors", "year"],
        trendline='ols',
        trendline_options=trendline_options
    )
    fig.update_layout(layout)
    fig.update_layout(
        xaxis=dict(title=x),
        yaxis=dict(title=y)
    )
    # fig.update_traces(line_width=4)
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    fig.update_traces(selector=dict(mode="lines"), line=dict(dash="dash", color="#d81275", width=5), name="Trend")
    fig.show()


def plot_directors(df, x, y, trendline_options=None, hover_data=None):
    if not trendline_options:
        trendline_options = {}

    fig = px.scatter(
        df, 
        x=x, 
        y=y, 
        color_discrete_sequence=["#79b8b8"], 
        hover_data=hover_data,
        trendline='ols',
        trendline_options=trendline_options
    )
    fig.update_layout(layout)
    fig.update_layout(
        xaxis=dict(title=x),
        yaxis=dict(title=y)
    )
    # fig.update_traces(line_width=4)
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    fig.update_traces(selector=dict(mode="lines"), line=dict(dash="dash", color="#d81275", width=5), name="Trend")
    fig.show()


def create_heatmap_array(df, bins=500, sigma=25):
    # 1. Create 2D Histogram
    heatmap, xedges, yedges = np.histogram2d(
        df['norm_y'], df['norm_x'], 
        bins=bins, range=[[0, 1], [0, 1]]
    )
    
    # 2. Apply Gaussian Blur to smooth out the "dots"
    heatmap = ndimage.gaussian_filter(heatmap, sigma=sigma)
    
    # 3. Normalize so the entire map sums to 1 (Probability Density)
    return heatmap / np.sum(heatmap)


def get_global_baseline(engine, sample_size=10000):
    with engine.connect() as conn:
        is_mysql = 'mysql' in str(engine.url)

        if is_mysql:
    # MySQL uses MD5 and CONV to turn hex to decimal
    # We take the first 6 chars of MD5 (~16 million possibilities)
            hash_logic = "CONV(LEFT(MD5(CONCAT(ff.imdb_id, ff.frame_num, ff.face_num)), 6), 16, 10) % 100 = 0"
        else:
            # DuckDB uses || for concat and ::BIGINT for casting
            hash_logic = "('0x' || LEFT(MD5(CONCAT(ff.imdb_id, ff.frame_num, ff.face_num)), 6))::BIGINT % 100 = 0"

        query = f"""
            SELECT 
                (ff.x1 + ff.x2) / (2.0 * di.img_width) as norm_x,
                (ff.y1 + ff.y2) / (2.0 * di.img_height) as norm_y
            FROM factFace ff
            JOIN dimDetectionInfo di ON ff.work_id = di.work_id
            WHERE {hash_logic}
            LIMIT {sample_size}
        """
        params = {"sample_size": sample_size}
        global_baseline_df = pd.read_sql_query(db.text(query), conn, params=params)
    return global_baseline_df
    

def get_global_baseline_widescreen(engine, sample_size=10000):
    # We use CRC32 because it's much faster than MD5 for simple sampling
    # It turns the 'movie+frame+face' combo into a number, then we pick a slice of those numbers
    with engine.connect() as conn:
        query = """
            SELECT 
                (ff.x1 + ff.x2) / (2.0 * di.img_width) as norm_x,
                (ff.y1 + ff.y2) / (2.0 * di.img_height) as norm_y
            FROM factFace ff
            JOIN dimDetectionInfo di ON ff.work_id = di.work_id
            JOIN dimWork dw ON ff.work_id = dw.work_id
            WHERE (ff.work_id % 100) = 0
                AND dw.year >= 1953
            LIMIT :sample_size
        """
        params = {"sample_size": sample_size}
        global_baseline_df = pd.read_sql_query(db.text(query), conn, params=params)
    return global_baseline_df
    

def get_global_baseline_top_1(engine, sample_size=10000):
    # We use CRC32 because it's much faster than MD5 for simple sampling
    # It turns the 'movie+frame+face' combo into a number, then we pick a slice of those numbers
    with engine.connect() as conn:
        query = """
            SELECT 
                (ff.x1 + ff.x2) / (2.0 * di.img_width) as norm_x, 
                (ff.y1 + ff.y2) / (2.0 * di.img_height) as norm_y
            FROM factFace ff
            JOIN dimDetectionInfo di ON ff.work_id = di.work_id
            JOIN dimWork dw ON ff.work_id = dw.work_id
            WHERE HASH(ff.imdb_id, ff.frame_num, ff.face_num) % 100 = 0
                AND ff.is_largest_face
            LIMIT :sample_size
        """
        params = {"sample_size": sample_size}
        global_baseline_df = pd.read_sql_query(db.text(query), conn, params=params)
    return global_baseline_df


def get_global_baseline_grid(engine):
    # We use CRC32 because it's much faster than MD5 for simple sampling
    # It turns the 'movie+frame+face' combo into a number, then we pick a slice of those numbers
    with engine.connect() as conn:
        query = """
            SELECT
                -- Top Row
                AVG(pct_faces_top_left) AS avg_top_left,
                AVG(pct_faces_top_center) AS avg_top_center,
                AVG(pct_faces_top_right) AS avg_top_right,
                
                -- Middle Row
                AVG(pct_faces_mid_left) AS avg_middle_left,
                AVG(pct_faces_mid_center) AS avg_middle_center,
                AVG(pct_faces_mid_right) AS avg_middle_right,
                
                -- Bottom Row
                AVG(pct_faces_bot_left) AS avg_bottom_left,
                AVG(pct_faces_bot_center) AS avg_bottom_center,
                AVG(pct_faces_bot_right) AS avg_bottom_right
            FROM factWork
        """
        global_baseline_df = pd.read_sql_query(db.text(query), conn)
    return global_baseline_df
    

def plot_comparison(residual, title="The Welles Signature: Deviations from Hollywood Norms"):
    # Calculate the symmetric limit so 0 is exactly in the center of the color scale
    limit = max(abs(residual.min()), abs(residual.max()))
    
    fig = px.imshow(
        residual,
        # 'RdBu_r' equivalent in Plotly is 'RdBu' (it is already reversed by default)
        color_continuous_scale='RdBu_r', 
        range_color=[-limit, limit],
        labels=dict(x="Horizontal Position", y="Vertical Position", color="Deviation"),
        x=np.linspace(0, 1, residual.shape[1]),
        y=np.linspace(0, 1, residual.shape[0]),
        title=title,
        aspect="auto"
    )

    # Configure the 'Camera coordinate' layout
    fig.update_layout(layout)
    fig.update_layout(
        xaxis=dict(title="Horizontal Screen Position", range=[0, 1]),
        # Inverting Y to match standard image/video top-down coordinates
        yaxis=dict(title="Vertical Screen Position", range=[1, 0]),
        coloraxis_colorbar=dict(title="Deviation"),
        # template="plotly_dark",
        width=800,
        height=700
    )

    # Add a zero-line to the colorbar or annotations if needed
    fig.show()


def plot_global_average(df, title="The 'Hollywood Bullseye': Industry Average Framing"):
    # 1. Generate the heatmap array 
    avg_map = create_heatmap_array(df)
    
    # 2. Create the base Heatmap
    fig = px.imshow(
        avg_map,
        labels=dict(x="Horizontal Position", y="Vertical Position", color="Density"),
        x=np.linspace(0, 1, avg_map.shape[1]),
        y=np.linspace(0, 1, avg_map.shape[0]),
        color_continuous_scale='Turbo',
        title=title,
        aspect="auto"
    )

    # 3. Add Rule of Thirds Overlay (using shapes)
    rule_of_thirds_lines = []
    for pos in [1/3, 2/3]:
        # Vertical lines
        rule_of_thirds_lines.append(dict(
            type="line", x0=pos, x1=pos, y0=0, y1=1,
            line=dict(color="white", width=2, dash="dash"), opacity=0.4
        ))
        # Horizontal lines
        rule_of_thirds_lines.append(dict(
            type="line", x0=0, x1=1, y0=pos, y1=pos,
            line=dict(color="white", width=2, dash="dash"), opacity=0.4
        ))

    # 4. Update Layout for the "Camera View" feel
    fig.update_layout(layout)
    
    fig.update_layout(
        shapes=rule_of_thirds_lines,
        xaxis=dict(range=[0, 1], title="Normalized Width"),
        yaxis=dict(range=[1, 0], title="Normalized Height"), # Invert Y to match image coordinates
        width=800,
        height=800,
        # template="plotly_dark" # Dark theme looks great for film data
    )

    fig.show()


def compare_film_to_sample_heat(title, engine):  
    with engine.connect() as conn:  
        imdb_id = conn.execute(
            db.text("""
                SELECT
                    imdb_id
                FROM dimWork 
                WHERE title LIKE :title
            """), 
            {"title": title}
        ).fetchone()[0]

        global_baseline_df = get_global_baseline(engine)
        
        query = """
            SELECT (ff.x1 + ff.x2) / (2.0 * di.img_width) as norm_x, 
                (ff.y1 + ff.y2) / (2.0 * di.img_height) as norm_y
            FROM factFace ff
            JOIN dimDetectionInfo di ON ff.work_id = di.work_id
            WHERE ff.imdb_id = :imdb_id
        """
        ck_df = pd.read_sql_query(db.text(query), conn, params={"imdb_id": imdb_id})

    standard_map = create_heatmap_array(global_baseline_df)
    ck_map = create_heatmap_array(ck_df)

    residual = ck_map - standard_map
    plot_comparison(residual)


def compare_film_to_sample_grid(title, engine, plot_title=None):
    avg_grid_norm = get_sample_grid(engine)

    imdb_id = imdb_from_title(title, engine)
    with engine.connect() as conn:
        query = """
            SELECT
                pct_tl,
                pct_tc,
                pct_tr,
                pct_ml,
                pct_mc,
                pct_mr,
                pct_bl,
                pct_bc,
                pct_br
            FROM factWork
            WHERE imdb_id = :imdb_id
        """
        df = pd.read_sql_query(db.text(query), conn, params={"imdb_id": imdb_id})
    title_grid = np.array(df.values).reshape(3, 3)

    diff_grid = title_grid - avg_grid_norm
    plot_grid(diff_grid, plot_title=plot_title)