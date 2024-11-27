from ast import literal_eval
import numpy as np
import pandas as pd


def createBoundingBoxFeatures(df):
    # parse auxiliary features
    df[['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max']] = pd.DataFrame(df['diagnostics_Mask-original_BoundingBox'].apply(literal_eval).tolist() , index=df.index)

    df['width'] = df['x_max'] - df['x_min']
    df['height'] = df['y_max'] - df['y_min']
    df['depth'] = df['z_max'] - df['z_min']

    # new features
    df['new_bbox_volume'] = df['width'] * df['height'] * df['depth']
    df['new_bbox_x_center'] = (df['x_min'] + df['x_max']) / 2
    df['new_bbox_y_center'] = (df['y_min'] + df['y_max']) / 2
    df['new_bbox_z_center'] = (df['z_min'] + df['z_max']) / 2
    df['new_bbox_aspect_ratio_xy'] = df['width'] / df['height']
    df['new_bbox_aspect_ratio_xz'] = df['width'] / df['depth']
    df['new_bbox_aspect_ratio_yz'] = df['height'] / df['depth']
    df['new_bbox_diagonal'] = np.sqrt((df['x_max'] - df['x_min'])**2 + 
                         (df['y_max'] - df['y_min'])**2 + 
                         (df['z_max'] - df['z_min'])**2)
    
    # clean excess
    df.drop(['diagnostics_Mask-original_BoundingBox'],axis=1,inplace=True)
    df.drop(['x_min', 'y_min','z_min', 'x_max', 'y_max', 'z_max'],axis=1,inplace=True)
    df.drop(['width', 'height','depth'],axis=1,inplace=True)

    return df    

def createCenterOfMassFeatures(df):
    df[['x','y','z']] =  pd.DataFrame(df['diagnostics_Mask-original_CenterOfMass'].apply(literal_eval).tolist() , index=df.index)
    df['new_CenterOfMass_Magnitude'] = np.sqrt( df['x']**2 + df['y']**2 + df['z']**2)

    df.drop(['x','y','z','diagnostics_Mask-original_CenterOfMass'],axis=1,inplace=True)

    df[['x','y','z']] =  pd.DataFrame(df['diagnostics_Mask-original_CenterOfMassIndex'].apply(literal_eval).tolist() , index=df.index)
    df['new_CenterOfMassIndex_Magnitude'] = np.sqrt( df['x']**2 + df['y']**2 + df['z']**2)

    df.drop(['x','y','z','diagnostics_Mask-original_CenterOfMassIndex'],axis=1,inplace=True)
    return df

def preprocessingV1(df):

    y = df['Transition']
    X = df.drop('Transition', axis = 1)

    ## Dealing with object features
    # Experiment new features
    X = createBoundingBoxFeatures(X)
    X = createCenterOfMassFeatures(X)
    # Delete useless object features
    X.drop(X.select_dtypes(include='object').drop(columns=['Transition'], errors='ignore').columns, axis=1, inplace=True)

    # Drop features with always the same value. 159
    features_without_variance = X.columns[X.nunique()==1].to_list()
    X.drop(columns=features_without_variance, axis=1, inplace = True)

    return X,y


    