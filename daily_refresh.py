from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash
from flask import send_from_directory
import time
import datetime
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import warnings
from os import listdir
import plotly.graph_objs as go

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pd.options.mode.chained_assignment = None
import dash_table
def get_sectors(expiry):
    
    # Url with anything selected in dropdown
    url = "https://www.moneycontrol.com/stocks/marketstats/open-interest/sector/futures-banking_financial_services-" + expiry + ".html?classic=true"
    
    # Load the page and parse
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'lxml')
    
    select = soup.find('select', {'name':'sel_sect', 'id':'sel_sect'})
    options = select.findAll('option')
    
    sectors = []
    sectors_link = []
    for i in options:
        sectors_link.append(i['value'])
        sectors.append(i.text)
        
    sectors_link = sectors_link[1:]
    sectors = sectors[1:]
    
    return sectors, sectors_link


def get_table(sectors, sectors_link):
    
    all_data = pd.DataFrame()
    
    for cnt in range(len(sectors)):
        
        # Url of specific sector
        url = "https://www.moneycontrol.com/stocks/marketstats/open-interest/" + sectors_link[cnt] + "?classic=true"
        
        # Load the page and parse
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'lxml')
        
        table = soup.find('table', {'class':'advdec'})
        table_headers = table.findAll('th')
        
        headers = []
        for header in table_headers:
            headers.append(header.text)
            
        rows = table.findAll('tr')
        table_content = table.find_all('td')
        content = [ele.text for ele in table_content]
        
        data_rows = [content[x:x+len(headers)] for x in range(0, len(content), len(headers))]
        
        if len(data_rows) > 0:
            df = pd.DataFrame(data_rows[:-1])
            df.columns = headers
            df['Sector'] = sectors[cnt]
            start1 = sectors_link[cnt].find('futures-') + 8
            end1 = sectors_link[cnt].find('.html')
            start2 = sectors_link[cnt][start1:end1].find('-') + 1
            df['Expiry'] = sectors_link[cnt][start1:end1][start2:]

            all_data = all_data.append(df, ignore_index=True)        
    
    if len(all_data.columns == 11):
        all_data.columns.values[2] = "% Chg - CMP"
        all_data.columns.values[5] = "% Chg - OI"
        all_data.columns.values[8] = "% Chg - Volume"
        
        all_data.columns.values[4] = "Change - OI"
        all_data.columns.values[7] = "Change - Volume"
        
    else:
        print("Got different number of columns than expected")
    
    all_data.loc[:,'% Chg - CMP'] = all_data.loc[:,'% Chg - CMP'].str.replace('%','').str.replace('+','').astype('float')
    all_data.loc[:,'% Chg - OI'] = all_data.loc[:,'% Chg - OI'].str.replace('%','').str.replace(',','').str.replace('+','').astype('float')
    all_data.loc[:,'% Chg - Volume'] = all_data.loc[:,'% Chg - Volume'].str.replace('%','').str.replace(',','').str.replace('+','').astype('float')
    all_data.loc[:,'CMP'] = all_data.loc[:,'CMP'].str.replace(',','').astype('float')
    all_data.loc[:,'Open Interest'] = all_data.loc[:,'Open Interest'].str.replace(',','').astype('int')
    all_data.loc[:,'Volume'] = all_data.loc[:,'Volume'].str.replace(',','').astype('int')
    all_data.loc[:,'Change - OI'] = all_data.loc[:,'Change - OI'].str.replace(',','').astype('int')
    all_data.loc[:,'Change - Volume'] = all_data.loc[:,'Change - Volume'].str.replace(',','').astype('int')
    
    return all_data


def get_agg_data():
    
    sectors_near, sectors_link_near = get_sectors('near')
    sectors_next, sectors_link_next = get_sectors('next')
    sectors_far, sectors_link_far = get_sectors('far')
    
    OI_data_near = get_table(sectors_near, sectors_link_near)
    OI_data_next = get_table(sectors_next, sectors_link_next)
    OI_data_far = get_table(sectors_far, sectors_link_far)
    
    OI_data = pd.DataFrame()
    OI_data = OI_data.append(OI_data_near).append(OI_data_next).append(OI_data_far)
    
    aggregated = {'Open Interest': sum,
              'Change - OI': sum,
              'Volume': sum,
              'Change - Volume': sum}
    
    cumulative_metrics = OI_data.groupby('Company').agg(aggregated)
    cumulative_metrics['OI Change %'] = cumulative_metrics['Change - OI'] / cumulative_metrics['Open Interest'] * 100
    cumulative_metrics.columns = ['Cumulative OI', 'Cumulative OI Change', 'Cumulative Volume', 'Cumulative Volume Change', 'Cumulative OI Change %']
    
    OI_data_moneycontrol = OI_data_near.join(cumulative_metrics, on='Company')
    OI_data_moneycontrol = OI_data_moneycontrol.loc[:,OI_data_moneycontrol.columns.isin(['Company', 'Sector', 'CMP', '% Chg - CMP', 'Cumulative OI', 'Cumulative OI Change %'])]
    
    cols = OI_data_moneycontrol.columns.tolist()
    cols = cols[0::3] + cols[1:3:] + cols[4:6:]
    OI_data_moneycontrol = OI_data_moneycontrol[cols]
    OI_data_moneycontrol = OI_data_moneycontrol.rename(columns = {'Company':'Scrip'})
    
    return OI_data_moneycontrol

def generate_dataset(scrip_sector_mapping, scrip_list, exclusion_list = []):

    dataset = pd.DataFrame()
    cnt = 0

    for file in listdir('bhavcopy_data/100'):
        filename = 'bhavcopy_data/100/' + file
        df = pd.read_csv(filename, sep = ',', header=-1)
        df.columns = ['Scrip', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']
        df['Date'] = pd.to_datetime(df['Date'].astype('str'), format = '%Y%m%d')
        df['Expiry_Num'] = df['Scrip'].apply(lambda x: x[(x.rfind('-')+1):])
        df['Expiry'] = np.where(df['Expiry_Num'] == 'I', 'near', np.where(df['Expiry_Num'] == 'II', 'next', 'far'))
        df['Scrip'] = df['Scrip'].apply(lambda x: x[:x.rfind('-')])
        df = df.loc[~df['Scrip'].isin(['DJIA', 'FTSE100', 'S&P500']),:]
        df = df.drop('Expiry_Num',axis=1)
    
        df_near = df.loc[df['Expiry'] == 'near',:]
    
        aggregated = {'Volume': sum,
                      'OI': sum
                     }
    
        cumulative_metrics = df.groupby('Scrip').agg(aggregated)
        cumulative_metrics.columns = ['Cumulative Volume', 'Cumulative OI']

        df_agg = df_near.join(cumulative_metrics, on='Scrip')
        df_agg = df_agg.drop(['Volume', 'OI', 'Expiry'], axis=1)

        dataset = dataset.append(df_agg, ignore_index=True)
        
        #cnt = cnt+1
        #print(cnt, file)
                   
    dataset = dataset.loc[dataset['Scrip'].isin(scrip_list),:]
    dataset = dataset.sort_values(['Scrip', 'Date']).reset_index(drop=True)
    
    dataset['Price Change %'] = dataset.groupby('Scrip')['Close'].apply(lambda x: x.pct_change())
    dataset['Price Change %'] = dataset['Price Change %'].apply(lambda x: x*100)

    dataset['OI Change %'] = dataset.groupby('Scrip')['Cumulative OI'].apply(lambda x: x.pct_change())
    dataset['OI Change %'] = dataset['OI Change %'].apply(lambda x: x*100)
    
    dataset['OI Interpretation'] = np.where(dataset['Price Change %'] > 0, 
                                        np.where(dataset['OI Change %'] > 0, "Long Buildup", "Short Covering"), 
                                        np.where(dataset['OI Change %'] > 0, "Short Buildup", "Long Unwinding"))
    
    dataset = pd.merge(dataset, scrip_sector_mapping, on='Scrip', how='inner')
    dataset = dataset.loc[~dataset['Scrip'].isin(exclusion_list),:]
    
    return dataset

def generate_table(dataframe, max_rows=200):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

OI_data_moneycontrol = get_agg_data()
scrip_sector_mapping = OI_data_moneycontrol.loc[:,['Scrip','Sector']]
scrip_list = OI_data_moneycontrol.loc[:,'Scrip']
exclusion_list = pd.read_csv('exclusion_list.csv')
exclusion_list = list(exclusion_list.iloc[:,0])
OI_hist = generate_dataset(scrip_sector_mapping,scrip_list, exclusion_list)
OI_hist.to_csv('OI_hist.csv', index=False)
OI_data_moneycontrol.to_csv('OI_data_moneycontrol.csv', index=False)
app = dash.Dash()
app.title = 'Daily Dashboard'
app.config.supress_callback_exceptions = True
start_date_daily = str(datetime.datetime.now().strftime("%d-%m-%Y"))
end_date_daily = str(datetime.datetime.now().strftime("%d-%m-%Y"))

external_css = [
    'https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0-rc.2/css/materialize.min.css',
    'https://fonts.googleapis.com/icon?family=Material+Icons',
    'https://codepen.io/muhnot/pen/bKzaZr.css'
]

external_js = [
     'https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0-rc.2/js/materialize.min.js'
]

for my_js in external_js:
  app.scripts.append_script({"external_url": my_js})


for css in external_css:
    app.css.append_css({"external_url": css})


#loaded navbar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(className='navbar-fixed',children=[
        html.Nav([
            html.Div(className='nav-wrapper grey darken-4',children=[
                dcc.Link(className='brand-logo left',children=[html.I(className='material-icons left',children=['blur_on']),'Daily Dashboard'],href='/daily_layout')
            ])
        ])
    ]),
    html.Br(),
    #display container, layouts are returned to this container
    html.Div(className='container',children=[
        html.Div(className='row',children=[

            html.Div(className='input-field col l4 m3 s12',children=[
                html.I(className="material-icons prefix",children=['date_range']),
                dcc.Input(id='inputddate-1', type='text',value=start_date_daily),
                html.Label(className='active' ,htmlFor='inputddate-1',children= ['Start Date'])
            ]),

            html.Div(className='input-field col l4 m3 s12',children=[
                html.I(className="material-icons prefix",children=['date_range']),
                dcc.Input(id='inputddate-2', type='text',value=end_date_daily),
                html.Label(className='active', htmlFor='inputddate-2',children= ['End Date'])
            ]),
            html.Div(className='input-field col l4 m3 s12',children=[
                html.I(className="material-icons prefix",children=['dehaze']),
                dcc.Input(id='n_rows', type='text',value=10),
                html.Label(className='active', htmlFor='n_rows',children= ['No. of Rows'])
            ])
        ]),


        html.Div(className='row',children=[
            html.Div(className='input-field col l3 m3 s12',children=[
                html.I(className="material-icons prefix",children=['expand_more']),
                dcc.Input(id='gt_oi', type='text',value=100),
                html.Label(className='active', htmlFor='gt_oi',children= ['OI <'])
            ]),

        html.Div(className='input-field col l3 m3 s12',children=[
            html.I(className="material-icons prefix",children=['expand_less']),
            dcc.Input(id='lt_oi', type='text',value=0),
            html.Label(className='active', htmlFor='lt_oi',children= ['OI >'])
            ]),

        html.Div(className='input-field col l3 m3 s12',children=[
            html.I(className="material-icons prefix",children=['expand_more']),
            dcc.Input(id='gt_price', type='text',value=100),
            html.Label(className='active', htmlFor='gt_price',children= ['Price <'])
            ]),

        html.Div(className='input-field col l3 m3 s12',children=[
            html.I(className="material-icons prefix",children=['expand_less']),
            dcc.Input(id='lt_price', type='text',value=0),
            html.Label(className='active', htmlFor='lt_price',children= ['Price >'])
            ]),
        html.Div(className='col s6 m3 l4',children=[
                html.H5(className='blue-text',children=['Sector: ']),
                dcc.Dropdown(
                    id='sector',
                    options=[
                        {'label':'All','value':'All'},
                        {'label':'Automotive','value':'Automotive'},
                        {'label':'Banking & Financial Services','value':'Banking & Financial Services'},
                        {'label':'Cement & Construction','value':'Cement & Construction'},
                        {'label':'Chemicals','value':'Chemicals'},
                        {'label':'Conglomerates','value':'Conglomerates'},
                        {'label':'Consumer Non-durables','value':'Consumer Non-durables'},
                        {'label':'Engineering & Capital','value':'Engineering & Capital'},
                        {'label':'Food & Beverages','value':'Food & Beverages'},
                        {'label':'Information Technology','value':'Information Technology'},
                        {'label':'Manufacturing','value':'Manufacturing'},
                        {'label':'Media & Entertainment','value':'Media & Entertainment'},
                        {'label':'Metals & Mining','value':'Metals & Mining'},
                        {'label':'Miscellaneous','value':'Miscellaneous'},
                        {'label':'Oil & Gas','value':'Oil & Gas'},
                        {'label':'Pharmaceuticals','value':'Pharmaceuticals'},
                        {'label':'Retail & Real Estate','value':'Retail & Real Estate'},
                        {'label':'Services','value':'Services'},
                        {'label':'Telecommunication','value':'Telecommunication'},
                        {'label':'Tobacco','value':'Tobacco'},
                        {'label':'Utilities','value':'Utilities'}
                    ],
                    value='All'
                    ) 
             ]),
            html.Div(className='input-field col l4 m3 s12',children=[
                html.Center(html.Button(className='waves-effect waves-light btn', type=['Submit'], children=['View'], id='my-button-daily'))
            ]),

        html.H5(className='blue-text',children=['Scrip: ']),
        html.Div(className='input-field col l4 m3 s12',children=[
            html.I(className="material-icons prefix",children=['expand_less']),
            dcc.Input(id='scrip', type='text',value='All'),
            html.Label(className='active', htmlFor='scrip',children= ['Scrip'])
        ]),

        ])
        ]),
        html.Hr(),
        html.Div(className='input-field col l4 m3 s12',children=[
                html.Center(html.Button(className='waves-effect waves-light btn', children=['Refresh'], type=['Submit'], id='refresh'))
            ]),
        html.Div(id='ref_out',style={'display':'block'},children=[]),
        html.Hr(),
        html.Div(id='daily_dashboard',style={'display':'block'},children=[])
])
@app.callback(
    Output('daily_dashboard', 'children'),
    [Input('my-button-daily','n_clicks')],
    [State('inputddate-1', 'value'),State('inputddate-2', 'value'),State('n_rows', 'value'),State('lt_oi', 'value'),State('gt_oi', 'value'),State('lt_price', 'value'),State('gt_price', 'value'),State('sector', 'value'),State('scrip', 'value')]
    )
def update_output(n_clicks,start_date,end_date,n,lt_oi,gt_oi,lt_price,gt_price,sector,scrip):
    if n_clicks or n_clicks>=1:
        n = int(n)
        lt_oi =int(lt_oi)
        gt_oi = int(gt_oi)
        lt_price =int(lt_price)
        gt_price = int(gt_price)
        if lt_oi==None:
            lt_oi = 100
        if gt_oi==None:
            gt_oi = 0
        if lt_price==None:
            lt_price = 100
        if gt_price==None:
            gt_price = 0

        if lt_oi>gt_oi:
            lt_oi = 100
            gt_oi = 0
        
        if lt_price>gt_price:
            lt_price = 100
            gt_price = 0
        OI_hist = pd.read_csv('OI_hist.csv')
        OI_hist['Date'] = pd.to_datetime(OI_hist['Date'])
        OI_data_moneycontrol = pd.read_csv('OI_data_moneycontrol.csv')
        sday,smonth,syear = int(str(start_date).split('-')[0]),int(str(start_date).split('-')[1]),int(str(start_date).split('-')[2])
        eday,emonth,eyear = int(str(end_date).split('-')[0]),int(str(end_date).split('-')[1]),int(str(end_date).split('-')[2])
        start_date = datetime.datetime(syear,smonth,sday)
        end_date = datetime.datetime(eyear,emonth,eday)
        OI_hist_filtered = OI_hist.loc[np.logical_and(OI_hist['Date'] >= start_date, OI_hist['Date'] <= end_date),:]
        OI_hist_filtered.loc[:,'OI Percentile'] = OI_hist_filtered.groupby('Scrip', group_keys=False)['OI Change %'].rank(pct=True).mul(100).round(2)
        OI_hist_filtered.loc[:,'Price Percentile'] = OI_hist_filtered.groupby('Scrip', group_keys=False)['Price Change %'].rank(pct=True).mul(100).round(2)
        if(len(OI_hist_filtered)):
            OI_hist_filtered = OI_hist_filtered.loc[OI_hist_filtered['Date'] == max(OI_hist_filtered['Date']),:].reset_index(drop=True)

        if sector !='All':
            OI_hist_filtered = OI_hist_filtered[OI_hist_filtered['Sector']==sector]
        if scrip!='All':
            OI_hist_filtered = OI_hist_filtered[OI_hist_filtered['Scrip']==scrip]
        OI_hist_filtered1 = OI_hist_filtered[(OI_hist_filtered['Price Percentile'] >= lt_price) & (OI_hist_filtered['Price Percentile'] <=gt_price)]
        OI_hist_filtered2 = OI_hist_filtered[(OI_hist_filtered['OI Percentile'] >= lt_oi) & (OI_hist_filtered['OI Percentile'] <=gt_oi)]

        OI_latest2 = OI_hist_filtered2.sort_values('OI Percentile').reset_index(drop=True)
        OI_percentile_low = OI_latest2.loc[:,['Scrip','OI Percentile','Date','Sector']].head(n)
        OI_latest2 = OI_hist_filtered2.sort_values('OI Percentile', ascending = False).reset_index(drop=True)
        OI_percentile_high = OI_latest2.loc[:,['Scrip','OI Percentile','Date','Sector']].head(n)
        OI_latest1 = OI_hist_filtered1.sort_values('Price Percentile').reset_index(drop=True)
        price_percentile_low = OI_latest1.loc[:,['Scrip','Price Percentile','Date','Sector']].head(n)
        OI_latest1 = OI_hist_filtered1.sort_values('Price Percentile', ascending = False).reset_index(drop=True)
        price_percentile_high = OI_latest1.loc[:,['Scrip','Price Percentile','Date','Sector']].head(n)
        OI_hist_c = OI_hist_filtered[(OI_hist_filtered['Price Percentile'] >= lt_price) & (OI_hist_filtered['Price Percentile'] <=gt_price)]
        OI_hist_c1 = OI_hist_c[(OI_hist_c['OI Percentile'] >= lt_oi) & (OI_hist_c['OI Percentile'] <=gt_oi)]
        OI_hist_c = OI_hist_c1.loc[:,['Scrip','Price Percentile','OI Percentile','Date']].head(n)
        OI_hist_c = OI_hist_c1.loc[:,['Scrip','Price Percentile','OI Percentile','Date']].head(n)
        
        merged = OI_hist.loc[:,['Scrip','Date','Close','Cumulative OI']]
        merged.sort_values(['Date', 'Scrip'])
        merged['Price Change %'] = merged.groupby('Scrip')['Close'].apply(lambda x: x.pct_change())
        merged['Price Change %'] = merged['Price Change %'].apply(lambda x: x*100)

        merged['OI Change %'] = merged.groupby('Scrip')['Cumulative OI'].apply(lambda x: x.pct_change())
        merged['OI Change %'] = merged['OI Change %'].apply(lambda x: x*100)

        filtered = merged.loc[np.logical_and(merged['Date'] >= start_date, merged['Date'] <= end_date),:]

        filtered = filtered.dropna()

        filtered.loc[:,'OI Percentile'] = filtered.groupby('Scrip')['OI Change %'].rank(pct=True).mul(100)
        filtered.loc[:,'Price Percentile'] = filtered.groupby('Scrip', group_keys=False)['Price Change %'].rank(pct=True).mul(100)
        

        change = filtered.loc[np.logical_and(np.logical_and(filtered['Scrip']==scrip, 
                                                                     filtered['Date'] >= start_date),
                                                      filtered['Date'] <= end_date),:].reset_index(drop=True)

        
        layout1 = html.Div(className='',children=[
                    html.Div(className='row',children=[
                    html.Div(className='col s12 m6 l6 z-depth-2',children=[
                            html.H5(className='blue-text',children=['Top OI Percentile: ']),
                            html.Div([generate_table(OI_percentile_high)])        
                        ]),
                    html.Div(className='col s12 m6 l6 z-depth-1',children=[
                        html.H5(className='blue-text',children=['Least OI Percentile: ']),
                        html.Div([generate_table(OI_percentile_low)])
                        ]),
                    html.Hr(),
                    html.Div(className='col s12 m6 l6 z-depth-1',children=[
                        html.H5(className='blue-text',children=['Top Price Percentile: ']),
                        html.Div([generate_table(price_percentile_high)])
                        ]),
                    html.Div(className='col s12 m6 l6 z-depth-2',children=[
                        html.H5(className='blue-text',children=['Least Price Percentile: ']),
                        html.Div([generate_table(price_percentile_low)])
                        ]),
                    html.Hr(),
                    html.Div(className='col s12 m12 l12 z-depth-1',children=[
                        html.H5(className='blue-text',children=['Combined Price and OI: ']),
                        html.Div([generate_table(OI_hist_c)]),
                    ])
                    ])
                ])
        layout2 = html.Div(className='',children=[
                    html.Div(className='row',children=[
                    html.Div(className='col s12 m6 l6 z-depth-2',children=[
                            html.H5(className='blue-text',children=['Top OI Percentile: ']),
                            html.Div([generate_table(OI_percentile_high)])        
                        ]),
                    html.Div(className='col s12 m6 l6 z-depth-1',children=[
                        html.H5(className='blue-text',children=['Least OI Percentile: ']),
                        html.Div([generate_table(OI_percentile_low)])
                        ]),
                    html.Hr(),
                    html.Div(className='col s12 m6 l6 z-depth-2',children=[
                        html.H5(className='blue-text',children=['Top Price Percentile: ']),
                        html.Div([generate_table(price_percentile_high)])
                        ]),
                    html.Div(className='col s12 m6 l6 z-depth-1',children=[
                        html.H5(className='blue-text',children=['Least Price Percentile: ']),
                        html.Div([generate_table(price_percentile_low)])
                        ]),
                    html.Hr(),
                    html.Div(className='col s12 m12 l12 z-depth-1',children=[
                        html.H5(className='blue-text',children=['Combined Price and OI: ']),
                        html.Div([generate_table(OI_hist_c)]),
                    ])
                    ]),
                    html.Div(className='col s12 m7 l12',children=[
                        dcc.Graph(
                            figure=go.Figure(
                                data=[
                                    go.Line(
                                        x=change['Date'],
                                        y=change['Close'],
                                        name='Close of ' + str(scrip),
                                    ),
                                ],
                                layout=go.Layout(
                                    title='Close of ' + str(scrip),
                                    showlegend=True,
                                    legend=go.layout.Legend(
                                        x=0,
                                        y=1.0
                                    ),
                                    xaxis=dict(
                                        tickfont=dict(
                                            size=7,
                                            color='rgb(107, 107, 107)'
                                        )
                                    )
                                )
                            ),
                            id='my-graph1',
                            style={'height': '',
                                    'width': '100%'
                                   }
                        )
                        ]),
                    html.Div(className='col s12 m7 l12',children=[
                        dcc.Graph(
                            figure=go.Figure(
                                data=[
                                    go.Bar(
                                        x=change['Date'],
                                        y=change['Cumulative OI'],
                                        name='Cumulative OI of '+str(scrip),
                                    )
                                ],
                                layout=go.Layout(
                                    title='Cumulative OI of '+str(scrip),
                                    showlegend=True,
                                    legend=go.layout.Legend(
                                        x=0,
                                        y=1.0
                                    ),
                                    xaxis=dict(
                                        tickfont=dict(
                                            size=7,
                                            color='rgb(107, 107, 107)'
                                        )
                                    )
                                )
                            ),
                            id='my-graph',
                            style={'height': '',
                                    'width': '100%'
                                   }
                        )
                        ])

                ])

        if scrip=='All':
            return layout1
        else:
            return layout2

@app.callback(
    Output('ref_out', 'children'),
    [Input('refresh','n_clicks')])
def update(n):
    if n:
        OI_data_moneycontrol = get_agg_data()
        scrip_sector_mapping = OI_data_moneycontrol.loc[:,['Scrip','Sector']]
        scrip_list = OI_data_moneycontrol.loc[:,'Scrip']
        exclusion_list = pd.read_csv('exclusion_list.csv')
        exclusion_list = list(exclusion_list.iloc[:,0])
        OI_hist = generate_dataset(scrip_sector_mapping,scrip_list, exclusion_list)
        OI_hist.to_csv('OI_hist.csv', index=False)
        OI_data_moneycontrol.to_csv('OI_data_moneycontrol.csv', index=False)
        return html.H5('Data updated. ')
server = app.server

if __name__ == '__main__':
    app.run_server(host='localhost',port=9605)
