import vapoursynth as vs


# 绝大部分场景趋于理想
# 画面快速平移时平行线有概率会烂，svpflow_alt 无此问题
def svpflow(clip, fps, freq):
    super_params = "{pel:2,gpu:0,scale:{up:2,down:4}}"
    analyse_params = "{block:{w:32,h:32,overlap:2},main:{levels:4,search:{type:4,distance:-8,coarse:{type:4,distance:-5,bad:{range:0}}},penalty:{plevel:1.3,pzero:110,pnbour:75}},refine:[{thsad:200,search:{type:4,distance:2}}]}"
    smoothfps_params = "{rate:{num:" + str(freq) + \
        ",den:1000,abs:true},algo:21,mask:{area:100},scene:{limits:{m1:1800,m2:3600,scene:5200,zero:100,blocks:45}}}"

    svp_super = core.svp1.Super(clip, super_params)
    svp_analyse = core.svp1.Analyse(
        svp_super["clip"], svp_super["data"], clip, analyse_params)
    svp_smooth = core.svp2.SmoothFps(
        clip, svp_super["clip"], svp_super["data"], svp_analyse["clip"], svp_analyse["data"], smoothfps_params, src=clip, fps=fps)

    return svp_smooth


# 相较于 svpflow 会更加流畅一点，但烂帧肉眼可见增多
def svpflow_alt(clip, fps, freq):
    super_params = "{pel:2,gpu:0,scale:{up:2,down:4}}"
    analyse_params = "{block:{w:32,h:32,overlap:2},main:{levels:0,search:{type:4,distance:-8,coarse:{type:4,distance:-6,bad:{range:0}}},penalty:{plevel:1.3,pzero:110,pnbour:75}},refine:[{thsad:400,search:{type:4,distance:2}}]}"
    smoothfps_params = "{rate:{num:" + str(freq) + \
        ",den:1000,abs:true},algo:21,mask:{area:200},scene:{limits:{m1:1800,m2:3600,scene:5200,zero:100,blocks:45}}}"

    svp_super = core.svp1.Super(clip, super_params)
    svp_analyse = core.svp1.Analyse(
        svp_super["clip"], svp_super["data"], clip, analyse_params)
    svp_smooth = core.svp2.SmoothFps(
        clip, svp_super["clip"], svp_super["data"], svp_analyse["clip"], svp_analyse["data"], smoothfps_params, src=clip, fps=fps)

    return svp_smooth


# NVIDIA Optical Flow
def svpflow_nvof(clip, fps, freq):
    smoothfps_params = "{rate:{num:" + \
        str(freq) + ",den:1000,abs:true},algo:21}"

    crop_clip = clip.std.Crop(0, 0, 0, 2)
    svp_smooth = core.svp2.SmoothFps_NVOF(
        clip, smoothfps_params, nvof_src=crop_clip, src=clip, fps=fps)

    return svp_smooth


# mvtools-high
def mvtools(clip, fps, freq):
    clip = core.std.AssumeFPS(clip, fpsnum=fps, fpsden=1)

    mv_super = core.mv.Super(clip, pel=2, sharp=2, rfilter=4,
                             hpad=16, vpad=8, levels=0)
    mv_vec_b = core.mv.Analyse(mv_super, blksize=32, blksizev=16, overlap=16, overlapv=8, levels=0,
                               isb=True, search=3, searchparam=0, pelsearch=3, badrange=-1, badsad=10000)
    mv_vec_f = core.mv.Analyse(mv_super, blksize=32, blksizev=16, overlap=16, overlapv=8, levels=0,
                               isb=False, search=3, searchparam=0, pelsearch=3, badrange=-1, badsad=10000)
    mv_smooth = core.mv.BlockFPS(clip, mv_super, mv_vec_b, mv_vec_f,
                                 num=freq, den=1000, mode=2, ml=1020.0, thscd1=16320, thscd2=255)

    return mv_smooth


# scale-down
def clip_fit(clip, clip_w, clip_h, fit_w, fit_h):
    ratio = min(fit_w / clip_w, fit_h / clip_h)
    if (ratio < 1):
        clip = clip.resize.Lanczos(
            width=int(clip_w * ratio), height=int(clip_h * ratio))
    return clip


# 补帧方法 (svpflow / svpflow_alt / svpflow_nvof / mvtools)
clip_interp = svpflow
fps = container_fps if (container_fps > 0.1) else 24
freq = display_fps if (display_fps > 0.1) else 60

if (fps < freq):
    core = vs.core
    core.max_cache_size = 2048

    clip = video_in
    clip = clip.resize.Point(format=vs.YUV420P8, dither_type="random")
    clip = clip_fit(clip, video_in_dw, video_in_dh, 1920, 1080)
    clip = clip_interp(clip, fps, int(freq * 1000))
    clip = core.std.AssumeFPS(clip, fpsnum=clip.fps_num, fpsden=clip.fps_den)
    clip.set_output()
