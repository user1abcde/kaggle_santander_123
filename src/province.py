import pandas as pd


def get_province_data():
    province_area = pd.read_csv('static/province_area.csv')
    province_gdp = pd.read_csv('static/province_gdp.csv')
    province_pop = pd.read_csv('static/province_pop.csv')

    province_area.columns = province_area.columns.map(lambda s: s.replace(" ", ""))
    province_gdp.columns = province_gdp.columns.map(lambda s: s.replace(" ", ""))
    province_pop.columns = province_pop.columns.map(lambda s: s.replace(" ", ""))

    province_area['Province'] = province_area['Province'].str.upper().str.replace('^A ', "").str.replace('^EL ',
                                                                                                         "").str.replace(
        '^LA ', "").str.replace('^LAS ', "").str.replace("^ ", "").str.replace(" $", "")
    province_pop['Province'] = province_pop['Province'].str.upper().str.replace('^A ', "").str.replace('^EL ',
                                                                                                       "").str.replace(
        '^LA ', "").str.replace('^LAS ', "").str.replace("^ ", "").str.replace(" $", "")
    province_gdp['Province'] = province_gdp['Province'].str.upper().str.replace('^A ', "").str.replace('^EL ',
                                                                                                       "").str.replace(
        '^LA ', "").str.replace('^LAS ', "").str.replace("^ ", "").str.replace(" $", "")

    province_area['Province'] = province_area['Province'].str.replace('Á', 'A').str.replace('Ó', 'O').str.replace('Í',
                                                                                                                  'I').str.replace(
        'É', 'E').str.replace('È', 'E')
    province_pop['Province'] = province_pop['Province'].str.replace('Á', 'A').str.replace('Ó', 'O').str.replace('Í',
                                                                                                                'I').str.replace(
        'É', 'E').str.replace('È', 'E')
    province_gdp['Province'] = province_gdp['Province'].str.replace('Á', 'A').str.replace('Ó', 'O').str.replace('Í',
                                                                                                                'I').str.replace(
        'É', 'E').str.replace('È', 'E')

    province_area['Province'] = province_area['Province'].str.replace('ALACANT', 'ALICANTE').str.replace('CASTELLO',
                                                                                                         'CASTELLON').str.replace(
        'SEVILLE', 'SEVILLA')
    province_gdp['Province'] = province_gdp['Province'].str.replace('ALACANT', 'ALICANTE').str.replace('CASTELLONN',
                                                                                                       'CASTELLON').str.replace(
        'SEVILLE', 'SEVILLA')

    province_area['Province'] = province_area['Province'].str.replace('BALEARIC ISLANDS', 'BALEARS').str.replace(
        'BISCAY', 'BIZKAIA').str.replace('LLEIDA', 'LERIDA').str.replace('NAVARRE', 'NAVARRA')
    province_pop['Province'] = province_pop['Province'].str.replace('BALEARIC ISLANDS', 'BALEARS').str.replace('BISCAY',
                                                                                                               'BIZKAIA').str.replace(
        'LLEIDA', 'LERIDA').str.replace('NAVARRE', 'NAVARRA')
    province_gdp['Province'] = province_gdp['Province'].str.replace('BALEARIC ISLANDS', 'BALEARS').str.replace('BISCAY',
                                                                                                               'BIZKAIA').str.replace(
        'LLEIDA', 'LERIDA').str.replace('NAVARRE', 'NAVARRA')

    province_gdp['Province'] = province_gdp['Province'].str.replace('PALMA', 'PALMAS').str.replace('TENERIFE',
                                                                                                   'SANTA CRUZ DE TENERIFE').str.replace(
        'MALLORCA', 'BALEARS')

    province_area.set_index("Province", drop=True, inplace=True)
    province_pop.set_index("Province", drop=True, inplace=True)
    province_gdp.set_index("Province", drop=True, inplace=True)

    province_df = pd.DataFrame()
    province_df['density'] = province_pop['Population'] / province_area['Area']
    province_df['gdp'] = province_gdp['GDP']

    return province_df
