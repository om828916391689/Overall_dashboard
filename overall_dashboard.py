import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder


# Load data
df = pd.read_csv('student_data_full (2).csv')

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Sidebar filters
st.sidebar.title("Filters")
branch_filter = st.sidebar.multiselect("Select Branch", options=df['Branch'].unique(), default=df['Branch'].unique())
year_filter = st.sidebar.multiselect("Select Year", options=df['Year'].unique(), default=df['Year'].unique())
cgpa_range = st.sidebar.slider("CGPA Range", float(df['CGPA'].min()), float(df['CGPA'].max()), (float(df['CGPA'].min()), float(df['CGPA'].max())))
attendance_range = st.sidebar.slider("Attendance % Range", int(df['Attendance'].min()), int(df['Attendance'].max()), (int(df['Attendance'].min()), int(df['Attendance'].max())))

# Apply filters
filtered_df = df[
    (df['Branch'].isin(branch_filter)) &
    (df['Year'].isin(year_filter)) &
    (df['CGPA'] >= cgpa_range[0]) & (df['CGPA'] <= cgpa_range[1]) &
    (df['Attendance'] >= attendance_range[0]) & (df['Attendance'] <= attendance_range[1])
]

# Tabs
st.title("🎓 Student Performance Dashboard")
tabs = st.tabs(["Overview", "Performance Analysis", "Individual Profile", "Leaderboard", "Insights"])

# Tab 1: Overview
with tabs[0]:
    st.title("🏆 Student Performance Overview")

    # KPIs
    total_students = len(df)
    avg_cgpa = df['CGPA'].mean()
    avg_attendance = df['Attendance'].mean()
    total_branches = df['Branch'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="🎓 Total Students", value=total_students)
    col2.metric(label="📊 Average CGPA", value=f"{avg_cgpa:.2f}")
    col3.metric(label="✅ Average Attendance (%)", value=f"{avg_attendance:.1f}%")
    col4.metric(label="🏢 Number of Branches", value=total_branches)

    st.markdown("---")

    # Bar chart: Students per Branch
    st.subheader("Student Distribution by Branch")
    branch_counts = df['Branch'].value_counts().reset_index()
    branch_counts.columns = ['Branch', 'Count']
    fig_branch_dist = px.bar(
        branch_counts, x='Count', y='Branch',
        orientation='h',
        text='Count',
        color='Count',
        color_continuous_scale='Viridis',
        title="Number of Students in Each Branch"
    )
    fig_branch_dist.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
    st.plotly_chart(fig_branch_dist, use_container_width=True)

    st.markdown("---")

    # Pie chart: Student Count by Year
    st.subheader("Student Count by Academic Year")
    year_counts = df['Year'].value_counts().reset_index()
    year_counts.columns = ['Year', 'Count']
    fig_year_pie = px.pie(
        year_counts, values='Count', names='Year',
        title="Distribution of Students Across Academic Years",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_year_pie, use_container_width=True)

    st.markdown("---")

    # Histogram: CGPA distribution
    st.subheader("CGPA Distribution")
    fig_cgpa_hist = px.histogram(
        df, x='CGPA', nbins=20,
        title="Distribution of CGPA among Students",
        color_discrete_sequence=['#636EFA']
    )
    st.plotly_chart(fig_cgpa_hist, use_container_width=True)

    # Histogram: Attendance distribution
    st.subheader("Attendance Distribution")
    fig_attendance_hist = px.histogram(
        df, x='Attendance', nbins=20,
        title="Distribution of Attendance (%) among Students",
        color_discrete_sequence=['#EF553B']
    )
    st.plotly_chart(fig_attendance_hist, use_container_width=True)

    st.markdown("---")

    # Box plot: CGPA by Branch
    st.subheader("CGPA Distribution by Branch")
    fig_cgpa_box = px.box(
        df, x='Branch', y='CGPA',
        title="CGPA Variation Across Branches",
        color='Branch',
        points="all"
    )
    st.plotly_chart(fig_cgpa_box, use_container_width=True)



    # Average attendance & CGPA progress bars
    st.subheader("Average Attendance & CGPA Progress")

    def progress_bar(label, value, max_val=10):
        pct = value / max_val
        st.markdown(f"**{label}:** {value:.2f} / {max_val}")
        st.progress(min(pct, 1.0))

    col5, col6 = st.columns(2)
    with col5:
        progress_bar("Average CGPA", avg_cgpa, max_val=10)
    with col6:
        # Attendance scaled from 0-100 to 0-10 for progress bar
        progress_bar("Average Attendance (%)", avg_attendance / 10, max_val=10)

    st.markdown("---")

    st.markdown("#### Quick Insights")
    st.info(
        f"- Total branches: **{total_branches}**\n"
        f"- Average CGPA is **{avg_cgpa:.2f}**, showing overall academic health.\n"
        f"- Attendance average is **{avg_attendance:.1f}%**, indicating class participation.\n"
        f"- Branch with most students: **{branch_counts.iloc[0]['Branch']}** ({branch_counts.iloc[0]['Count']})"
    )

# Tab 2: Performance Analysis
with tabs[1]:
    st.plotly_chart(px.box(filtered_df, x='Branch', y='CGPA', title="CGPA by Branch"))
    st.plotly_chart(px.violin(filtered_df, x='Year', y='Attendance', box=True, points='all', title="Attendance % by Year"))
    st.plotly_chart(px.scatter(filtered_df, x='Attendance', y='CGPA', color='Branch', trendline='ols', title="Attendance vs CGPA"))

    fig, ax = plt.subplots()
    sns.histplot(filtered_df['IA Marks'], kde=True, ax=ax)
    st.pyplot(fig)

# Tab 3: Individual Profile
with tabs[2]:
    st.title("👤 Individual Student Performance")

    student_names = filtered_df['Name'].unique()
    selected_student = st.selectbox("Select Student", student_names)
    student_data = filtered_df[filtered_df['Name'] == selected_student].iloc[0]

    # Profile Details with markdown styling
    st.markdown(f"""
    ### 🎓 Profile: **{selected_student}**

    - **Branch:** `{student_data['Branch']}`
    - **Year:** `{student_data['Year']}`
    - **CGPA:** `{student_data['CGPA']:.2f}`
    - **Attendance:** `{student_data['Attendance']}%`
    - **IA Marks:** `{student_data['IA Marks']}`
    - **Skills:** `{student_data['Skills']}`
    - **Interest:** `{student_data['Interest']}`
    """)

    # Rank Calculation
    branch_students = filtered_df[
        (filtered_df['Branch'] == student_data['Branch']) &
        (filtered_df['Year'] == student_data['Year'])
    ].sort_values(by='CGPA', ascending=False).reset_index(drop=True)

    rank = branch_students[branch_students['Name'] == selected_student].index[0] + 1
    total_students = len(branch_students)

    st.markdown(f"### 🏅 Rank in {student_data['Branch']} Year {student_data['Year']}: **{rank} / {total_students}**")

    # Radar Chart
    categories = ['CGPA', 'Attendance', 'IA Marks']
    student_values = [
        student_data['CGPA'],
        student_data['Attendance'],
        student_data['IA Marks']
    ]

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
          r=student_values,
          theta=categories,
          fill='toself',
          name=selected_student,
          line_color='royalblue'
    ))

    fig_radar.update_layout(
      polar=dict(
        radialaxis=dict(visible=True, range=[0, max(max(student_values)+1, 10)])
      ),
      showlegend=False,
      title=f"Performance Radar Chart for {selected_student}"
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    # Improvement Suggestions
    st.markdown("### 💡 Suggestions")

    suggestions = []

    if student_data['Attendance'] < 75:
        suggestions.append("⚠️ **Attendance below 75%. Needs immediate improvement to meet eligibility criteria.**")

    if student_data['CGPA'] < 6.0:
        suggestions.append("📈 **CGPA below 6.0. Recommend academic mentoring and extra assignments.**")

    if str(student_data['Skills']).strip().lower() in ['none', 'na', '']:
        suggestions.append("💡 **No skills listed. Encourage participation in workshops, contests, or soft skills training.**")

    if str(student_data['Interest']).strip().lower() in ['none', 'na', '']:
        suggestions.append("📝 **No specific interest mentioned. Suggest career counseling or aptitude testing.**")

    if len(suggestions) == 0:
        st.success("✅ Student is performing well across all tracked parameters!")

    else:
        for suggestion in suggestions:
            st.warning(suggestion)








# Tab 4: Leaderboard
with tabs[3]:
    st.title("🏆 Leaderboard")

    if filtered_df.empty:
        st.warning("No data available for Leaderboard.")
    else:
        # Copy & sort
        df_leaderboard = filtered_df.copy()
        df_leaderboard = df_leaderboard.sort_values(by='CGPA', ascending=False).reset_index(drop=True)
        df_leaderboard['Rank'] = df_leaderboard.index + 1

        # Filters for Branch and Year
        branches = df_leaderboard['Branch'].unique()
        years = sorted(df_leaderboard['Year'].unique())

        selected_branch = st.selectbox("Filter by Branch", options=['All'] + list(branches))
        selected_year = st.selectbox("Filter by Year", options=['All'] + list(years))

        # Apply filters
        if selected_branch != 'All':
            df_leaderboard = df_leaderboard[df_leaderboard['Branch'] == selected_branch]
        if selected_year != 'All':
            df_leaderboard = df_leaderboard[df_leaderboard['Year'] == selected_year]

        # Add your job suitability code here:
        # ----------------------------------
        job_profiles = {
            "Data Scientist": ["python", "machine learning", "data analysis", "statistics"],
            "Web Developer": ["html", "css", "javascript", "react", "frontend", "backend"],
            "Cybersecurity Analyst": ["security", "networking", "cybersecurity", "ethical hacking"],
            "Software Engineer": ["java", "c++", "oop", "algorithms", "software development"],
            "Project Manager": ["management", "leadership", "planning", "communication"],
            "AI Researcher": ["ai", "deep learning", "neural networks", "research"],
            "Business Analyst": ["business", "analytics", "communication", "excel"],
        }

        def find_suitable_jobs(skills_str, interest_str):
            skills = str(skills_str).lower()
            interest = str(interest_str).lower()
            suitable_jobs = []
            for job, keywords in job_profiles.items():
                if any(k in skills for k in keywords) or any(k in interest for k in keywords):
                    suitable_jobs.append(job)
            return ", ".join(suitable_jobs) if suitable_jobs else "None"

        df_leaderboard['Suitable Jobs'] = df_leaderboard.apply(
            lambda row: find_suitable_jobs(row['Skills'], row['Interest']),
            axis=1
        )
        # ----------------------------------

        # Summary cards for CGPA and Attendance
        avg_cgpa = df_leaderboard['CGPA'].mean()
        avg_attendance = df_leaderboard['Attendance'].mean()
        col1, col2 = st.columns(2)
        col1.metric("Average CGPA", f"{avg_cgpa:.2f}")
        col2.metric("Average Attendance (%)", f"{avg_attendance:.2f}%")

        # AG Grid table with Suitable Jobs included
        gb = GridOptionsBuilder.from_dataframe(
            df_leaderboard[['Rank', 'Name', 'Branch', 'Year', 'CGPA', 'Attendance', 'Suitable Jobs']]
        )
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_default_column(editable=False, filter=True, sortable=True)
        grid_options = gb.build()

        AgGrid(df_leaderboard, gridOptions=grid_options, height=400)

        # Top 10 CGPA bar chart
        top_10 = df_leaderboard.head(10)
        st.subheader("Top 10 Students by CGPA")
        fig_bar = px.bar(top_10, x='Name', y='CGPA', color='CGPA',
                         color_continuous_scale='Viridis', title="Top 10 CGPA")
        st.plotly_chart(fig_bar, use_container_width=True)

        # CGPA Distribution histogram
        st.subheader("CGPA Distribution")
        fig_hist = px.histogram(df_leaderboard, x='CGPA', nbins=20, title='CGPA Distribution')
        st.plotly_chart(fig_hist, use_container_width=True)

        # Job suitability distribution chart
        st.subheader("Job Suitability Distribution")

        job_counts = {}
        for jobs in df_leaderboard['Suitable Jobs']:
            for job in jobs.split(", "):
                if job != "None":
                    job_counts[job] = job_counts.get(job, 0) + 1

        if job_counts:
            job_dist_df = pd.DataFrame({
                'Job': list(job_counts.keys()),
                'Count': list(job_counts.values())
            }).sort_values('Count', ascending=False)

            fig_jobs = px.bar(job_dist_df, x='Job', y='Count', title="Number of Students Suitable for Each Job")
            st.plotly_chart(fig_jobs, use_container_width=True)
        else:
            st.info("No suitable jobs found based on Skills and Interest.")


with tabs[4]:
    st.title("📊 Insights & Analytics")

    if df.empty:
        st.warning("No data available for insights.")
    else:
        st.subheader("Average CGPA Over Years by Branch")

        avg_cgpa_branch_year = df.groupby(['Year', 'Branch'])['CGPA'].mean().reset_index()

        fig_trend_branch = px.line(
            avg_cgpa_branch_year,
            x='Year',
            y='CGPA',
            color='Branch',
            markers=True,
            title="Average CGPA Trend by Year for Each Branch",
            labels={"CGPA": "Average CGPA", "Year": "Academic Year", "Branch": "Branch"}
        )

        st.plotly_chart(fig_trend_branch, use_container_width=True)


        # 2. Attendance vs CGPA correlation heatmap
        st.subheader("Correlation Heatmap")
        corr = df[['CGPA', 'Attendance', 'IA Marks']].corr()
        fig_corr = px.imshow(
            corr, text_auto=True, color_continuous_scale='Blues',
            title="Correlation Matrix between CGPA, Attendance and IA Marks"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # 3. At-risk students (CGPA < 6 and Attendance < 70%)
        st.subheader("At-risk Students (CGPA < 6 and Attendance < 70%)")
        at_risk = df[(df['CGPA'] < 6) & (df['Attendance'] < 70)]
        if at_risk.empty:
            st.info("No students are currently at risk.")
        else:
            st.dataframe(at_risk[['Name', 'Branch', 'Year', 'CGPA', 'Attendance']])

        # 4. Skill gap radar chart
        st.subheader("Skill Gap Analysis")

        # Define key skills to check for (customize based on your dataset)
        skill_columns = ['python', 'java', 'machine learning', 'html', 'css', 'cybersecurity', 'sql', 'react', 'excel']

        # Calculate average skill proficiency (simple presence count normalized 0-10 scale)
        avg_skills = {}
        skills_lower = df['Skills'].str.lower()
        for skill in skill_columns:
            avg_skills[skill] = skills_lower.str.contains(skill).mean() * 10

        import plotly.graph_objects as go
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=list(avg_skills.values()),
            theta=list(avg_skills.keys()),
            fill='toself',
            name='Average Skill Proficiency'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=False,
            title="Average Skill Proficiency Radar Chart"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # 5. Interest distribution pie chart
        st.subheader("Student Interest Distribution")
        interests = df['Interest'].str.lower().value_counts().reset_index()
        interests.columns = ['Interest', 'Count']
        fig_interest = px.pie(
            interests, values='Count', names='Interest',
            title="Distribution of Student Interests"
        )
        st.plotly_chart(fig_interest, use_container_width=True)

