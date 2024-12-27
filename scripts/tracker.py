def update_track(all_tracks, unmatches_track, offline_all_tracks, kf):
    '''
    predict(kalman) for all tracks before measuring new states

    Check for age of tracks and delete them from active tracks
    Store matched old tracks in offline_all_tracks
    Delete unmatched tracks older than 3
    '''
    all_tracks = all_tracks + unmatches_track 

    for n, obj in enumerate(all_tracks):
        if obj.status=='new' and obj.counter>3:
            del all_tracks[n]

        if obj.status=='matched' and obj.counter>30:
            offline_all_tracks.append(all_tracks[n])
            del all_tracks[n]

        pred_m, pred_c = kf.predict(obj.mean[-1], obj.covariance)
        obj.mean.append(pred_m)
        obj.covariance = pred_c
        obj.inc_count()

    return all_tracks, offline_all_tracks