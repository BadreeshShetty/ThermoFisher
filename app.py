import streamlit as st


st.set_page_config(
    page_title="Thermo Fisher Forecasting",
    page_icon="📈",
    layout="wide"
)

st.sidebar.success("Select the Forecasting approach above.")
st.image('Thermo_front_page.png', caption='Thermo Fisher Forecasting')

st.header("Abstract")
st.write("This paper presents a detailed analysis of monthly sales forecasting for the Biosciences Division of Thermo-Fisher \
         in the North American region, employing advanced machine learning and time series forecasting techniques. \
         The primary focus was on developing reliable forecast models to predict future sales trends. Key forecasting metrics,\
          including Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE), \
         were calculated and analyzed to assess the accuracy and reliability of the models.\
          A thorough comparison of the monthly sales forecast against the Annual Operating Plan (AOP) was conducted to identify any\
          significant variances or discrepancies. This comparison helped in determining the alignment of the forecast with the company's \
         established annual objectives and indicated whether any adjustments were necessary. This project pinpointed specific regions or \
         countries within North America where performance deviated from the AOP or forecasted sales, highlighting areas in need of strategic \
         focus. Based on the insights gained from the forecast and performance analysis, actionable recommendations were provided to address \
         the identified issues and optimize sales performance in the Biosciences Division.")

st.header("Introduction")
st.write("Thermo Fisher Scientific is a Biotechnology company based in Waltham, Massachusetts.\
          Thermo Fisher was formed through the merger of Thermo Electron and Fisher Scientific in 2006.\
          In the competitive and dynamic field of biotechnology, Thermo Fisher Scientific's Biosciences Division \
         faces the challenge of optimizing its sales strategies to maximize revenue and maintain market leadership. \
         With a diverse range of products and services catering to pharmaceutical, biotech, academic, and government sectors, \
         the division must navigate complex market dynamics and varying customer needs. The division's ability to accurately forecast \
         sales is crucial for effective resource allocation, strategic planning, and risk mitigation.")


st.header("Problem Overview")
st.write("The task at hand is to develop a sales forecasting model for the North America region, utilizing historical sales, \
         customer, and product data, along with relevant external market and economic indicators. This model must not only predict \
         sales for the upcoming month with high accuracy but also align with the division/'s Annual Operating Plan (AOP). \
         The challenge lies in effectively analyzing past trends and incorporating multifaceted data to create a predictive model \
         that can adapt to market shifts and customer behavior change. The effectiveness and reliability of the forecast model will be \
         evaluated using standard forecasting accuracy metrics such as Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and \
         Mean Absolute Percentage Error (MAPE). Success in this project lies in creating an accurate, data-driven forecast model that can\
          significantly inform strategic decisions and resource allocation, ultimately aiding the division in meeting or exceeding its sales\
          objectives. The AOP, a comprehensive document that was created by the Finance and Commercial leadership teams within Thermo-Fisher Scientific. \
         AOP is determined by looking at historical trends, any upcoming acquisitions, marketing trends, and any other factors which might influence sales. \
         These plans may be used to help allocate resources correctly, plan for strategic initiatives, manage strategy, and mitigate corporate risks. \
         Because of all the decisions that the plan is made, it is important to have an accurate plan for what will happen to sales throughout the quarter.")


st.header("Significance of Sales Forecasting")
st.write("Sales forecasting is an essential aspect of business management, serving as a fundamental tool in strategic planning and financial management. \
         It is a primary role in resource allocation and management, where accurate sales predictions guide the efficient distribution of resources like staffing, \
         inventory, and production capabilities. In this project we will use sales forecasting to identify potential market risks and you enhance customer relationship \
         management by providing insights into future sales trends, thereby allowing Thermo fisher BID division to anticipate customer needs and refine their marketing and service strategies. \
         In a competitive market, accurate and timely forecasts can offer a significant advantage enabling Thermo Fisher to quickly adapt to market changes and stay ahead of competitors. \
         It also supports strategic decision-making, helping the company identify growth opportunities and make informed decisions about market expansions or product diversification.")


st.header("Review of Existing Methods")
st.write("Thermo Fisher has historically employed a combination of traditional statistical methods, industry expertise, and the insights derived from the Annual Operating Plan (AOP) to formulate its sales projections. \
         However, as the market is always evolving, there are more and more things that can affect how much a company sells. Because of this, we need to take another look at the ways we predict sales to make sure they still \
         work well with all these changes and complexities. Understanding the nuances of the existing forecasting methodologies, including the reliance on the AOP, is crucial for identifying areas of improvement. It involves a \
         thorough examination of the algorithms, data sources, and assumptions underpinning the current forecasting model, as well as an exploration of how effectively the AOP encapsulates the multifaceted influences on sales.\
         In summary, the problem at hand is not merely a numerical misalignment between projections and outcomes; it is a strategic imperative for Thermo Fisher to recalibrate its forecasting mechanisms. \
         This recalibration involves harnessing the power of advanced analytics, machine learning, and time series analysis to construct a forecasting model that is not only more accurate in its predictions but also agile enough to adapt to the dynamic landscape in which Thermo Fisher operates.")

st.header('Dataset Description')

# Dataset Overview
st.write("""
The dataset in question is a quarterly sales record spanning from January 2022 to August 2023, encapsulating seven quarters of transactional data.
""")

# Dataset Characteristics
st.subheader('Dataset Characteristics')
st.write("""
Initially, the complete dataset comprised approximately 7.8 million rows, encompassing extensive customer interactions. However, for the purpose of this project, the dataset has been refined to include only the top 8 customers by sales volume, offering a focused view on the highest-impact sales activities. Additionally, the dataset encompasses around 15,000 Stock Keeping Units (SKUs), providing a substantial yet manageable subset for detailed analysis. This curated dataset enables a more streamlined examination of sales patterns and customer behaviors, facilitating a clearer understanding of the business problem.
""")

# Time Series Analysis
st.subheader('Time Series Analysis')
st.write("""
For the advanced stages of time series analysis, the dataset was augmented with data from two additional quarters, specifically from January 2021 to December 2021. This expansion allows for a broader temporal analysis, enhancing the robustness of the forecasting models by integrating pre-pandemic sales trends and thereby offering a more comprehensive overview of the sales trajectory over time. The inclusion of these earlier quarters is instrumental in developing an accurate and reliable sales forecast by providing a longer historical context to identify and account for longer-term trends, cyclicality, and seasonality in the sales data.
""")

# Key Columns
st.subheader('Key Columns for Analysis')
st.write("""
- Year, Quarter, Month, Day, Week: These temporal columns are crucial for time series analysis, allowing for the assessment of sales trends over time and the identification of any seasonal patterns or cycles in the data.
- Product Type Summary, Product Family Summary, Product Line Summary: These columns categorize the products sold, which is essential for understanding the sales distribution across different product lines and for identifying which products are performing well or may require more focus.
- Division Region, User Region, Ship to Region, Ship from Country, End User Country: These geographical indicators provide insight into market penetration and regional sales performance, helping to pinpoint areas with high sales volume or regions where market share could be improved.
- CDM End Market Group: This field identifies the market segment of the end user, such as Diagnostics, Pharma & Biotech, or Academic & Government, enabling a targeted analysis of sales by market sector.
- Quantity, Amount: The quantity sold and the sales amount in dollars are fundamental for measuring sales performance, forecasting future sales, and planning inventory and production.
- Order Number: This is used for tracking individual sales transactions and could be linked to specific sales or forecasting models.
""")

# Conclusion
st.write("""
These columns will allow for a comprehensive analysis of sales data, aiding in the forecasting of future trends, evaluation of market strategies, and optimization of product lines in accordance with the company's operational objectives.
""")

st.header("Data Preprocessing")
st.write("In our data preparation process for sales forecasting, we began by grouping the entire dataset by reference date. This step was crucial to analyze sales over time and identify trends essential for predicting future sales. \
         To streamline the dataset, we then dropped 10 columns, removing irrelevant or redundant information to improve the focus and manageability of our forecasting model. Recognizing the unique sales patterns on weekends, \
         we adjusted the sales data accordingly: Saturday sales were added to Friday's figures, and Sunday sales were shifted to Monday, ensuring that the data accurately reflects actual sales activities.")

st.header("Data enrichment")
st.write("To add depth to our analysis, we integrated additional datasets, including flu data and stock information, offering insights into how health trends and economic factors impact sales and sentiment. For model training and validation,\
          we divided the data into two segments, allocating 15 months for training and 3 months for testing, a standard practice in predictive modeling that helps in evaluating the model's accuracy on unseen data. Lastly, we scaled the data, a critical \
         step to normalize the range of values, ensuring that the statistical model accurately interprets each variable without bias from differing scales. Each of these steps was meticulously executed to prepare the dataset for an effective and accurate sales forecasting model.")






 

