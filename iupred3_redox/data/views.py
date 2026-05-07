import json
import os
import re
import subprocess
import tempfile
import threading
from random import randint
from urllib.error import HTTPError
from time import gmtime, strftime
import requests

from bokeh.embed import components
from bokeh.layouts import layout
from bokeh.models import HoverTool, OpenURL, TapTool, ColumnDataSource, Legend, Label, Range1d
from bokeh.plotting import figure
from bokeh.resources import CDN
from django.http import JsonResponse
from django.shortcuts import render

from . import download_email_sender
from . import iupred2a
from . import multifasta_handler
from . import uniprot_sql_api

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')


################################
#                              #
#   DJANGO request functions   #
#                              #
################################

def index(request):
    """
    Index site
    :param request:
    :return:
    """
    return render(request, 'index.html')


def plot(request):
    """
    Generate the plots
    :param request:
    :return:
    """
    request.session.flush()
    try:
        input_accession = request.POST.get('accession').strip()
    except AttributeError:
        return render(request, 'index.html')
    # Read in possible input seq, strip header if it is a FASTA, and strip newlines
    inp_seq = "".join([x.strip() for x in request.POST.get('inp_seq').splitlines() if not x.startswith(">")])
    # Store the cache
    request.session['inp_seq'] = inp_seq
    request.session['iupred_type'] = request.POST.get('iupred_type')
    request.session["context"] = request.POST.get('context')
    if not request.POST.get('context_checker'):
        request.session["context"] = ""

    # If input is an accession
    if input_accession:
        try:
            fasta_object = uniprot_sql_api.GetFasta(input_accession)
            request.session["len"] = len(fasta_object.sequence())
            request.session['accession'] = fasta_object.accession()
            request.session['sequence'] = fasta_object.sequence()
            tmp_file = fasta_object.write()
            request.session['tempfile_name'] = tmp_file.name
            header = fasta_object.header()
        except ValueError:
            return render(request, 'index.html',
                          {'error_message': "{} was not found in UniProt!".format(input_accession), 'accession': None})

    # If input is a sequence
    else:
        # try:
        # fasta_object = uniprot_sql_api.GetFasta(inp_seq, inp_seq=True)
        # pdb_map = pdb(fasta_object.accession())
        # seq_len = len(fasta_object.sequence())
        # header = fasta_object.header()
        # tmp_file = fasta_object.write()
        # request.session['accession'] = fasta_object.accession()
        # except ValueError:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
        tmp_file.write(">INP_SEQ\n{}".format(inp_seq.upper()))
        tmp_file.close()
        request.session["len"] = len(inp_seq)
        request.session['accession'] = ""
        request.session['sequence'] = inp_seq
        request.session['tempfile_name'] = tmp_file.name
        header = "Input sequence"

    # Email sending
    if request.POST.get('email') and "myfile" not in request.FILES:
        return render(request, 'index.html',
                      {'error_message': "Please select a file to upload!", 'accession': None})

    # In case of uploaded multifasta
    if "myfile" in request.FILES:
        return multifasta_analysis(request)

    # Error handling
    if input_accession and inp_seq:
        return render(request, 'index.html',
                      {'error_message': "Error: Swissprot ID/AC OR sequence input", 'accession': None})
    if not input_accession and not inp_seq:
        return render(request, 'index.html',
                      {'error_message': "Error: No input given", 'accession': None})

    # Create plots
    main_plot, glob_text = gener_main_plot(request)
    pfam_boxes = pfam_plot(request, main_plot)

    # For some reason labels are dependent on other plots, create at the end
    if request.session['accession']:
        ptm_boxes = ptm_plot(request, main_plot)
        pdb_boxes = pdb_plot(request, main_plot)
        experimental_dis_boxes = experimental_disorder_plot(request, main_plot)
        pfam_boxes.add_layout(Label(x=-8, y=3, text='PFAM', x_units='screen', y_units='screen', text_font_style="bold",
                                    text_font_size="10pt"), "left")
        experimental_dis_boxes.add_layout(
            Label(x=2, y=3, text='EXP DIS', x_units='screen', y_units='screen', text_font_style="bold",
                  text_font_size="8pt"), "left")
        if request.session['context'] == 'redox':
            redox_boxes = redox_box_plot(request, main_plot)
            final_grid = [main_plot, pfam_boxes, ptm_boxes, pdb_boxes, experimental_dis_boxes, redox_boxes]
        else:
            final_grid = [main_plot, pfam_boxes, ptm_boxes, pdb_boxes, experimental_dis_boxes]
    else:
        if request.session['context'] == 'redox':
            pfam_boxes.add_layout(
                Label(x=-8, y=3, text='PFAM', x_units='screen', y_units='screen', text_font_style="bold",
                      text_font_size="10pt"), "left")
            redox_boxes = redox_box_plot(request, main_plot)
            final_grid = [main_plot, pfam_boxes, redox_boxes]
        else:
            pfam_boxes.add_layout(
                Label(x=5, y=3, text='PFAM', x_units='screen', y_units='screen', text_font_style="bold",
                      text_font_size="10pt"), "left")
            final_grid = [main_plot, pfam_boxes]

    grid = layout(final_grid)
    script, div = components(grid, CDN)
    os.unlink(tmp_file.name)

    # Fill out the session for the raw generation
    session = randint(1000000, 9999999)
    request.session[session] = {}
    for key, val in request.session.items():
        if key == session:
            continue
        request.session[session][key] = val
    log_headers = ['accession', 'context', 'context_checker', 'email', 'inp_seq', 'iupred_type', 'myfile']
    with open('{}/logs/{}.log'.format(DATA_DIR, strftime('%Y_%m')), 'a') as file_h:
        file_h.write("{},{},".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), get_client_ip(request)))
        for i in log_headers:
            if request.POST.get(i):
                file_h.write('{},'.format(re.sub(',', '', "|||".join(request.POST.get(i).splitlines()))))
            else:
                file_h.write('false,')
        file_h.write("false,false\n")
    return render(request, "plot.html",
                  {"the_script": script, "the_div": div, "glob_text": glob_text, "session": session,
                   "header": header, "accession": request.session["accession"], "input_sequence": inp_seq})


def raw(request, session=""):
    """
    Raw text representation
    :param request:
    :param session: Session ID
    :return:
    """
    header = """# IUPred2A: context-dependent prediction of protein disorder as a function of redox state and protein binding
# Balint Meszaros, Gabor Erdos, Zsuzsanna Dosztanyi
# Nucleic Acids Research 2018, Submitted\n"""
    if session:
        try:
            for key, val in request.session[session].items():
                if key == session:
                    continue
                request.session[key] = val
        except KeyError:
            return render(request, 'index.html',
                          {'error_message': "Session has expired!", 'accession': None})
    # This function is used by REST API, so generate results again
    if request.session["inp_seq"]:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
        tmp_file.write(">INP_SEQ\n{}".format(request.session['inp_seq']))
        tmp_file.close()
        sequence = request.session['inp_seq']
    else:
        try:
            _obj = uniprot_sql_api.GetFasta(request.session["accession"])
            tmp_file = _obj.write()
            sequence = _obj.sequence()
        except ValueError:
            return render(request, 'raw.html', {'text': "{} not found!".format(request.session["accession"])})
    res = header
    if request.session['context'] == "anchor":
        anc_data = iupred2a.anchor2(tmp_file.name)
        res += "# POS\tAMINO ACID\tIUPRED SCORE\tANCHOR SCORE\n"
        for idx, val in enumerate(iupred2a.iupred(tmp_file.name, request.session['iupred_type'])[0]):
            res += "{}\t{}\t{:.4f}\t{:.4f}\n".format(idx + 1, sequence[idx], val, anc_data[idx])

    elif request.session["context"] == "redox":
        if request.session["inp_seq"]:
            seq = request.session["inp_seq"]
        else:
            try:
                seq = uniprot_sql_api.GetFasta(request.session["accession"]).sequence()
            except ValueError:
                return render(request, 'raw.html', {'text': "{} not found!".format(request.session["accession"])})
        tmp_file2 = tempfile.NamedTemporaryFile(delete=False, mode="w")
        tmp_file2.write(">INP_SEQ\n{}".format(seq.replace("C", "S")))
        tmp_file2.close()
        iup_redox_data = iupred2a.iupred(tmp_file2.name, request.session['iupred_type'])[0]
        iupred_data = iupred2a.iupred(tmp_file.name, request.session['iupred_type'])[0]
        regions = get_redox_regions(iup_redox_data, iupred_data)
        res += "# POS\tAMINO ACID\tIUPRED2 REDOX PLUS\tIUPRED2 REDOX MINUS\tREDOX REGION\n"
        for idx, val in enumerate(iupred_data):
            plc = "{}\t{}\t{:.4f}\t{:.4f}\t0\n".format(idx + 1, sequence[idx], val, iup_redox_data[idx])
            for start, end in regions.items():
                if start <= idx <= end - 1:
                    plc = "{}\t{}\t{:.4f}\t{:.4f}\t1\n".format(idx + 1, sequence[idx], val, iup_redox_data[idx])
            res += plc
        os.unlink(tmp_file2.name)
    else:
        res += "# POS\tAMINO ACID\tIUPRED SCORE\n"
        for idx, val in enumerate(iupred2a.iupred(tmp_file.name, request.session['iupred_type'])[0]):
            res += "{}\t{}\t{:.4f}\n".format(idx + 1, sequence[idx], val)

    os.unlink(tmp_file.name)
    return render(request, 'raw.html', {'text': res})


def raw_json(request, session=""):
    """
    JSON formatted raw output
    :param request:
    :param session: Session ID
    :return:
    """
    header = """# IUPred2A: context-dependent prediction of protein disorder as a function of redox state and protein binding
# Balint Meszaros, Gabor Erdos, Zsuzsanna Dosztanyi
# Nucleic Acids Research 2018, Submitted\n"""
    if session:
        for key, val in request.session[session].items():
            if key == session:
                continue
            request.session[key] = val
    if request.session["inp_seq"]:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
        tmp_file.write(">INP_SEQ\n{}".format(request.session['inp_seq']))
        tmp_file.close()
        sequence = request.session['inp_seq']
    else:
        try:
            _obj = uniprot_sql_api.GetFasta(request.session["accession"])
            tmp_file = _obj.write()
            sequence = _obj.sequence()
        except ValueError:
            return render(request, 'raw.html', {'text': "{} not found!".format(request.session["accession"])})
    json_data = {'meta': header, 'seqence': sequence}
    if request.session['context'] == "anchor":
        json_data['iupred2'] = iupred2a.iupred(tmp_file.name, request.session['iupred_type'])[0]
        json_data['anchor2'] = iupred2a.anchor2(tmp_file.name)
    elif request.session["context"] == "redox":
        if request.session["inp_seq"]:
            seq = request.session["inp_seq"]
        else:
            try:
                seq = uniprot_sql_api.GetFasta(request.session["accession"]).sequence()
            except ValueError:
                return render(request, 'raw.html', {'text': "{} not found!".format(request.session["accession"])})
        tmp_file2 = tempfile.NamedTemporaryFile(delete=False, mode="w")
        tmp_file2.write(">INP_SEQ\n{}".format(seq.replace("C", "S")))
        tmp_file2.close()
        iupred_data = iupred2a.iupred(tmp_file.name, request.session['iupred_type'])[0]
        iup_redox_data = iupred2a.iupred(tmp_file2.name, request.session['iupred_type'])[0]
        json_data['iupred2_redox_plus'] = iupred_data
        json_data['iupred2_redox_minus'] = iup_redox_data
        json_data['redox_sensitive_regions'] = [[start + 1, end] for start, end in
                                                get_redox_regions(iup_redox_data, iupred_data).items()]
        os.unlink(tmp_file2.name)
    else:
        json_data['iupred2'] = iupred2a.iupred(tmp_file.name, request.session['iupred_type'])[0]
    os.unlink(tmp_file.name)
    if not json_data:
        json_data["meta"] = "Not a valid REST URL"
    return JsonResponse(json_data)


def rest(request, accession, iupred_type="", context=''):
    """
    Rest API handler
    :param request:
    :param accession: Accession
    :param iupred_type: Optional, in case other helper function gives it
    :return:
    """
    request.session['inp_seq'] = ""
    request.session["context"] = context
    request.session['accession'] = accession.split(".")[0]
    if not iupred_type:
        request.session['iupred_type'] = "long"
    else:
        request.session['iupred_type'] = iupred_type
    if accession.split(".")[-1] == "json":
        with open('{}/logs/{}.log'.format(DATA_DIR, strftime('%Y_%m')), 'a') as file_h:
            file_h.write("{},{},{},true,json".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), get_client_ip(request),
                                                     ",".join(
                                                         [accession.split(".")[0], 'false', 'false', 'false', 'false',
                                                          request.session['iupred_type'], 'false'])))
            file_h.write("\n")
        return raw_json(request)
    with open('{}/logs/{}.log'.format(DATA_DIR, strftime('%Y_%m')), 'a') as file_h:
        file_h.write("{},{},{},true,raw".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), get_client_ip(request),
                                                ",".join(
                                                    [accession.split(".")[0], 'false', 'false', 'false', 'false',
                                                     request.session['iupred_type'], 'false'])))
        file_h.write("\n")
    return raw(request)


def rest_typed(request, iup_type, accession):
    """
    REST API handler if IUPred mode is given
    :param request:
    :param iup_type: IUPred mode
    :param accession: Accession
    :return:
    """
    if iup_type.lower() not in ["long", "short", "glob", 'redox', 'anchor']:
        return render(request, 'raw.html', {
            'text': "Mode {} is not valid IUPred type!\nUse either 'long', 'short' or 'glob'!".format(iup_type)})
    if iup_type.lower() in ['redox', 'anchor']:
        return rest(request, accession, context=iup_type.lower())
    return rest(request, accession, iupred_type=iup_type.lower())


def help_site(request):
    return render(request, "help.html")


def examples(request):
    return render(request, "examples.html")


def new_features(request):
    return render(request, "new_features.html")


def statistics(request):
    return render(request, "statistics.html")


def links(request):
    return render(request, "links.html")


def download(request):
    return render(request, "download.html", {"error_message": ""})


def sample1(request):
    return render(request, 'index.html', {"sample1": "P04637"})


def sample2(request):
    return render(request, 'index.html', {"sample2": """>sp|Q14061|COX17_HUMAN Cytochrome c oxidase copper chaperone OS=Homo sapiens GN=COX17 PE=1 SV=2
MPGLVDSNPAPPESQEKKPLKPCCACPETKKARDACIIEKGEEHCGHLIEAHKECMRALG
FKI"""})


def license_text(request):
    return render(request, 'license.html')


def stats_login(request):
    return render(request, 'stats_login.html')


def stats(request, file_to_open=""):
    import hashlib
    import math
    pw = request.POST.get('password')
    if pw:
        request.session['pw'] = pw
    else:
        if 'pw' not in request.session:
            return render(request, 'stats_login.html')
        pw = request.session['pw']
    hsh = hashlib.sha256(pw.encode()).hexdigest()
    accepted_hashes = ['9888deb1fe96428b74e07bd6ec52c0c78ad3020fd7d220cf6f5425ed621aac95']
    if hsh not in accepted_hashes:
        return render(request, 'stats_login.html')
    ####
    logfiles = sorted([f for f in os.listdir('{}/logs/'.format(DATA_DIR)) if
                       os.path.isfile(os.path.join('{}/logs/'.format(DATA_DIR), f))])
    logfiles.remove('sended_to')
    header = ['TIME', 'IP', 'accession', 'context', 'context_checker', 'email', 'inp_seq', 'iupred_type', 'myfile',
              'rest', 'rest_type']
    if file_to_open:
        data = []
        with open('{}/logs/{}'.format(DATA_DIR, file_to_open), 'r') as file_h:
            for line in file_h:
                data.append([i.replace("|||", "\n").strip() for n, i in enumerate(line.split(','))])
        return render(request, 'stats.html', locals())
    else:
        all_submission = {x: 0 for x in header}
        all_submission['ip'] = set()
        montly_submissions = {}
        for flh in logfiles:
            data = []
            montly_submissions[flh.split('.')[0]] = {x: 0 for x in header}
            with open('{}/logs/{}'.format(DATA_DIR, flh)) as fn:
                for line in fn:
                    data.append([i.replace("|||", "\n").strip() for n, i in enumerate(line.split(','))])
                    all_submission['ip'].add(line.split(',')[1])
            for line in data:
                for idx, elem in enumerate(line):
                    if elem.strip():
                        try:
                            all_submission[header[idx]] += 1
                            montly_submissions[flh.split('.')[0]][header[idx]] += 1
                        except IndexError:
                            pass
                            # print(line)
            montly_submissions[flh.split('.')[0]]['iup_long'] = sum(
                [1 for x in data if x[header.index('iupred_type')] == 'long'])
            montly_submissions[flh.split('.')[0]]['iup_short'] = sum(
                [1 for x in data if x[header.index('iupred_type')] == 'short'])

            montly_submissions[flh.split('.')[0]]['anchor'] = sum(
                [1 for x in data if x[header.index('context')] == 'anchor'])
            montly_submissions[flh.split('.')[0]]['redox'] = sum(
                [1 for x in data if x[header.index('context')] == 'redox'])

            montly_submissions[flh.split('.')[0]]['context_checker'] = sum(
                [1 for x in data if x[header.index('context_checker')] == 'true'])
            montly_submissions[flh.split('.')[0]]['json'] = sum(
                [1 for x in data if x[header.index('rest_type')] == 'json'])
            montly_submissions[flh.split('.')[0]]['raw'] = sum(
                [1 for x in data if x[header.index('rest_type')] == 'raw'])

            all_submission['iup_long'] = sum([1 for x in data if x[header.index('iupred_type')] == 'long'])
            all_submission['anchor'] = sum([1 for x in data if x[header.index('context')] == 'anchor'])
            all_submission['context_checker'] = sum([1 for x in data if x[header.index('context_checker')] == 'true'])
            all_submission['json'] = sum([1 for x in data if x[header.index('rest_type')] == 'json'])

        plot_data = [['accession', 'inp_seq'], ['iup_long', 'iup_short'], ['anchor', 'redox'], ['raw', 'json']]
        colors = ["#4477AA", "#CC6677", "#DDCC77", "#117733"]
        plots = []

        for idx, outer in enumerate(plot_data):
            plc = []
            for plot_type in outer:
                plot_obj = figure(plot_width=600, plot_height=400, title=plot_type)
                plot_obj.toolbar.logo = None
                plot_obj.toolbar_location = None
                plot_obj.title.align = 'center'
                plot_obj.xaxis.ticker = [i for i in range(len(montly_submissions))]
                plot_obj.xaxis.major_label_overrides = {i: date for i, date in
                                                        enumerate(sorted(montly_submissions.keys()))}
                plot_obj.xaxis.major_label_orientation = math.pi / 4
                plot_obj.vbar(x=[i for i in range(len(montly_submissions))], width=0.8, bottom=0,
                              top=[montly_submissions[i][plot_type] for i in sorted(montly_submissions.keys())],
                              color=colors[idx])
                plc.append(plot_obj)
            plots.append(plc)
        grid = layout(plots, sizing_mode='scale_width')
        script, div = components(grid, CDN)
    locs = []
    for ip in all_submission['ip']:
        coords = get_ip_location(ip)
        if coords:
            locs.append([float(coords[0]), float(coords[1])])
    return render(request, 'stats.html', locals())


################################
#                              #
#        Plotting and          #
#      helper functions        #
#                              #
################################


def multifasta_analysis(request):
    email_addr = request.POST.get('email')
    if not re.match(r"^[A-Za-z0-9\.\+_-]+@[A-Za-z0-9\._-]+\.[a-zA-Z]*$", email_addr):
        return render(request, 'index.html',
                      {'error_message': "Email address is not vaild", 'accession': None})
    if request.FILES['myfile'].size / (1024 ** 2) > 200:
        return render(request, 'index.html',
                      {'error_message': "File size must not exceed 200mb!", 'accession': None})

    if not email_addr:
        return render(request, 'index.html',
                      {'error_message': "Please provide an email address!", 'accession': None})
    thr = threading.Thread(target=multifasta_handler.handle_uploaded_file,
                           args=(email_addr, request.FILES['myfile']),
                           kwargs={"mode": request.session['iupred_type'],
                                   "context": request.session["context"]})
    thr.setDaemon(True)
    thr.start()
    return render(request, 'index.html',
                  {
                      'error_message': "The requested analysis has started! We will send the results to the given email address as soon as they are done!",
                      'accession': None})


def gener_main_plot(request):
    iupred_result = iupred2a.iupred(request.session['tempfile_name'], request.session['iupred_type'])
    # Generate globular data
    glob_text = ""
    if request.session['tempfile_name'] and request.session['iupred_type'] == "glob":
        glob_text = iupred_result[1]
        request.session["glob_text"] = glob_text
    # Plot the main diagram
    plot_obj = figure(plot_width=1000, plot_height=300, y_range=(0, 1),
                      x_range=Range1d(0, request.session["len"] + 1,
                                      bounds=(-request.session["len"], request.session["len"] * 2)),
                      tools="reset,xpan,xwheel_zoom,save",
                      active_scroll="xwheel_zoom", active_drag="xpan",
                      y_axis_label='Score', x_axis_label='Position')
    # x_range=Range1d(1, seq_len, bounds=(1, seq_len)),
    plot_obj.yaxis.axis_label_text_font_style = "bold"
    plot_obj.xaxis.axis_label_text_font_style = "bold"

    plot_obj.toolbar.logo = None
    plot_obj.xgrid.grid_line_color = None
    plot_obj.line([-1, request.session["len"] + 1], [0.5, 0.5], color="black")

    legend_list = []
    loc = (850, 2)
    mute = None

    # IUPred is always shown
    x_values = iupred_result[0]
    iu = plot_obj.line([n + 1 for n, _ in enumerate(x_values)], x_values, color="red", muted_alpha=0, line_width=2)
    legend_list.append(('IUPred2', [iu]))

    # In case context is ANCHOR
    if request.session['context'] == 'anchor':
        x_values = iupred2a.anchor2(request.session['tempfile_name'])
        an = plot_obj.line([n + 1 for n, _ in enumerate(x_values)], x_values, color="blue", muted_alpha=0, line_width=2)
        legend_list.append(("ANCHOR2 ", [an]))
        loc = (725, 2)
        mute = 'mute'

    # In case of redox context a new tmp file and IUPred run is needed
    elif request.session['context'] == 'redox':
        tmp_file2 = tempfile.NamedTemporaryFile(delete=False, mode="w")
        tmp_file2.write(">INP_SEQ\n{}".format(request.session['sequence'].replace("C", "S")))
        tmp_file2.close()
        x_values_redox = iupred2a.iupred(tmp_file2.name, request.session['iupred_type'])[0]
        redox_plot = plot_obj.line([n + 1 for n, _ in enumerate(x_values_redox)], x_values_redox, color="#6d2c62",
                                   line_width=2)
        loc = (575, 2)
        # Color the redox regions
        redox_regions = get_redox_regions(x_values_redox, x_values)
        # Store redox regions for box plots
        request.session['redox_regions'] = redox_regions
        for start, end in redox_regions.items():
            patch_x = list(x_values_redox[start:end])
            band_x2 = [n + start + 1 for n, _ in enumerate(patch_x)] + [n + start + 1 for n, _ in enumerate(patch_x)][
                                                                       ::-1]
            band_y2 = [x - (x - x_values[n + start]) for n, x in enumerate(patch_x)] + patch_x[::-1]
            plot_obj.patch(band_x2, band_y2, color='#AA4499', fill_alpha=0.5, line_alpha=0)
        legend_list = [("Plus", [iu])]
        legend_list.append(("Minus", [redox_plot]))
        legend_list.append(("Redox sensitive disorder score  ", []))
        # plot_obj.add_layout(
        #     Label(x=0, y=0, text='TEST', x_units='screen', y_units='screen', text_font_style="bold", text_font_size="8pt"), "above")
        # legend_list = []
        if request.session['context'] == 'redox':
            os.unlink(tmp_file2.name)

    if mute:
        legend = Legend(items=legend_list[::-1], location=loc, orientation="horizontal", padding=0, margin=2,
                        border_line_color="white", click_policy=mute)
    else:
        legend = Legend(items=legend_list[::-1], location=loc, orientation="horizontal", padding=0, margin=2,
                        border_line_color="white")
    plot_obj.add_layout(legend, 'above')
    return plot_obj, glob_text


def pfam_plot(request, plot1):
    pf = pfam(request.session['tempfile_name'])
    plot_obj = figure(plot_width=1000, plot_height=30, y_range=(0, 1), x_range=plot1.x_range,
                      tools="tap,xpan,xwheel_zoom",
                      active_scroll="xwheel_zoom", active_drag="xpan")
    plot_obj.toolbar.logo = None
    plot_obj.toolbar_location = None
    plot_obj.outline_line_color = "white"
    plot_obj.xgrid.grid_line_color = None
    plot_obj.ygrid.grid_line_color = None
    plot_obj.xaxis.visible = False
    plot_obj.yaxis.visible = False
    plot_obj.line([-1, request.session["len"] + 1], [0.5, 0.5], color="black")
    x, y, lnk, width, color, range_nfo, type_nfo = [], [], [], [], [], [], []
    for annot in pf:
        if annot["type"] == "Family":
            color.append("#CC6677")
        elif annot["type"] == "Repeat":
            color.append("#DDCC77")
        elif annot["type"] == "Motif":
            color.append("#117733")
        elif annot["type"] == "Domain":
            color.append("#88CCEE")
        elif annot["type"] == "Disordered":
            color.append("#332288")
        type_nfo.append(annot["type"])
        x.append(((annot["real_start"] + annot["real_end"]) / 2))
        y.append(0.5)
        lnk.append(annot["name"])
        width.append(annot["real_end"] - annot["real_start"])
        range_nfo.append("{}-{}".format(annot["real_start"], annot["real_end"]))
    source = ColumnDataSource(
        data=dict(x=x, y=y, width=width, color=color, lnk=lnk, range_nfo=range_nfo, type_nfo=type_nfo))
    plot_obj.rect(x='x', y='y', width='width', height=0.65, name="boxes",
                  color='color', line_color="black", source=source,
                  nonselection_fill_alpha=1, nonselection_fill_color='color', nonselection_line_color="black",
                  nonselection_line_alpha=1)
    hover = HoverTool(names=["boxes"], point_policy='follow_mouse', tooltips=[
        ("Name", "@lnk"),
        ("Type", "@type_nfo"),
        ("Range", "@range_nfo")
    ])
    plot_obj.add_tools(hover)
    url = "http://pfam.xfam.org/family/@lnk/"
    taptool = plot_obj.select(type=TapTool)
    taptool.callback = OpenURL(url=url)
    taptool.names = ["boxes"]
    return plot_obj


def ptm_plot(request, plot1):
    phosphorilation, acetilation, metilation = ptm(request)
    plot_obj = figure(plot_width=1000, plot_height=60, y_range=(0, 2), x_range=plot1.x_range, tools="xpan,xwheel_zoom",
                      active_scroll="xwheel_zoom", active_drag="xpan")
    plot_obj.toolbar.logo = None
    plot_obj.toolbar_location = None
    plot_obj.outline_line_color = "white"
    plot_obj.xgrid.grid_line_color = None
    plot_obj.ygrid.grid_line_color = None
    plot_obj.xaxis.visible = False
    plot_obj.yaxis.visible = False
    plot_obj.line([-1, request.session["len"] + 1], [0.2, 0.2], color="black")
    plot_obj.line([-1, request.session["len"] + 1], [1.2, 1.2], color="black")
    plot_obj.add_layout(Label(x=-13, y=15, x_units='screen', y_units='screen', text='PTM', text_font_style="bold",
                              text_font_size="10pt"), "left")
    x_line, y_line, x_circle, y_circle, phos_info, color = [], [], [], [], [], []
    # Each PTM needs to be plotted differently
    for elem in phosphorilation:
        if elem["res"] == "S":
            color.append("#117733")
            phos_info.append("pS {}".format(elem["pos"]))

        elif elem["res"] == "T":
            color.append("#999933")
            phos_info.append("pT {}".format(elem["pos"]))

        elif elem["res"] == "Y":
            color.append("#CC6677")
            phos_info.append("pY {}".format(elem["pos"]))
        x_line.append([elem["pos"]] * 2)
        y_line.append([1.2, 1.7])
        x_circle.append(elem["pos"])
        y_circle.append(1.7)
    for elem in metilation:
        if elem in acetilation:
            continue
        color.append("#DDCC77")
        phos_info.append("me{} {}".format(elem["res"], elem["pos"]))
        x_line.append([elem["pos"]] * 2)
        y_line.append([0.2, 0.7])
        x_circle.append(elem["pos"])
        y_circle.append(0.7)
    for elem in acetilation:
        if elem in metilation:
            phos_info.append("ac/me{} {}".format(elem["res"], elem["pos"]))
            color.append("#882255")
        else:
            phos_info.append("ac{} {}".format(elem["res"], elem["pos"]))
            color.append("#332288")
        x_line.append([elem["pos"]] * 2)
        y_line.append([0.2, 0.7])
        x_circle.append(elem["pos"])
        y_circle.append(0.7)
    source = ColumnDataSource(data=dict(x=x_circle, y=y_circle, color=color, phos_info=phos_info))
    plot_obj.circle(x='x', y='y', size=7, color='color', name="boxes", source=source)
    source = ColumnDataSource(data=dict(x=x_line, y=y_line, color=color, phos_info=phos_info))
    plot_obj.multi_line(xs="x", ys="y", line_width=2, color="color", name="boxes", source=source)
    hover = HoverTool(names=["boxes"], point_policy='follow_mouse', tooltips=[
        ("PTM", "@phos_info"),
    ])
    plot_obj.add_tools(hover)
    return plot_obj


def pdb_plot(request, plot1):
    plot_obj = figure(plot_width=1000, plot_height=30, y_range=(0, 1), x_range=plot1.x_range, tools="xpan,xwheel_zoom",
                      active_scroll="xwheel_zoom", active_drag="xpan")
    plot_obj.toolbar.logo = None
    plot_obj.toolbar_location = None
    plot_obj.outline_line_color = "white"
    plot_obj.line([-1, request.session["len"] + 1], [0.5, 0.5], color="black")
    plot_obj.xgrid.grid_line_color = None
    plot_obj.ygrid.grid_line_color = None
    plot_obj.xaxis.visible = False
    plot_obj.yaxis.visible = False
    plot_obj.add_layout(Label(x=-13, y=3, text='PDB', x_units='screen', y_units='screen', text_font_style="bold",
                              text_font_size="10pt"), "left")
    x, y, lnk, width, color, range_nfo = [], [], [], [], [], []
    for num, (_, pos_dct) in enumerate(
            sorted(pdb(request.session['accession']).items(), key=lambda d: d[1]["start"], reverse=True)):
        # if pos_dct["end"] > seq_len:
        #     pos_dct["end"] = seq_len
        # color.append(palettes.Category20_20[num % 20])
        color.append("#44AA99")
        x.append(((pos_dct["start"] + pos_dct["end"]) / 2))
        y.append(0.5)
        lnk.append(", ".join(pos_dct["ids"]))
        width.append(pos_dct["end"] - pos_dct["start"])
        range_nfo.append("{}-{}".format(pos_dct["start"], pos_dct["end"]))
    source = ColumnDataSource(data=dict(x=x, y=y, width=width, color=color, lnk=lnk, range_nfo=range_nfo))
    plot_obj.rect(x='x', y='y', width='width', height=0.65, name="boxes",
                  color='color', line_color="black", source=source,
                  nonselection_fill_alpha=1, nonselection_fill_color='color', nonselection_line_color="black",
                  nonselection_line_alpha=1)
    hover = HoverTool(names=["boxes"], point_policy='follow_mouse', tooltips=[
        ("Entry", "@lnk"),
        ("Range", "@range_nfo")
    ])
    plot_obj.add_tools(hover)
    return plot_obj


def experimental_disorder_plot(request, plot1):
    exp_disorder_regions = experimental_disorder(request)
    plot_obj = figure(plot_width=1000, plot_height=30, y_range=(0, 1), x_range=plot1.x_range, tools="xpan,xwheel_zoom",
                      active_scroll="xwheel_zoom", active_drag="xpan")
    plot_obj.toolbar.logo = None
    plot_obj.toolbar_location = None
    plot_obj.outline_line_color = "white"
    plot_obj.xgrid.grid_line_color = None
    plot_obj.ygrid.grid_line_color = None
    plot_obj.xaxis.visible = False
    plot_obj.yaxis.visible = False
    plot_obj.line([-1, request.session["len"] + 1], [0.5, 0.5], color="black")
    x, y, identifier, width, color, range_nfo, type_nfo = [], [], [], [], [], [], []
    for dct in exp_disorder_regions:
        color.append("#cc0000")
        type_nfo.append(dct["type"])
        identifier.append(dct['id'])
        x.append(((dct["start"] + dct["end"]) / 2))
        y.append(0.5)
        width.append(dct["end"] - dct["start"])
        range_nfo.append("{}-{}".format(dct["start"], dct["end"]))
    source = ColumnDataSource(
        data=dict(x=x, y=y, width=width, color=color, identifier=identifier, range_nfo=range_nfo, type_nfo=type_nfo))
    plot_obj.rect(x='x', y='y', width='width', height=0.65, name="boxes",
                  color='color', line_color="black", source=source,
                  nonselection_fill_alpha=1, nonselection_fill_color='color', nonselection_line_color="black",
                  nonselection_line_alpha=1)
    hover = HoverTool(names=["boxes"], point_policy='follow_mouse', tooltips=[
        ("Database", "@type_nfo"),
        ("Identifier", "@identifier"),
        ("Range", "@range_nfo")
    ])
    plot_obj.add_tools(hover)
    return plot_obj


def redox_box_plot(request, plot1):
    plot_obj = figure(plot_width=1000, plot_height=30, y_range=(0, 1), x_range=plot1.x_range, tools="xpan,xwheel_zoom",
                      active_scroll="xwheel_zoom", active_drag="xpan")
    plot_obj.toolbar.logo = None
    plot_obj.toolbar_location = None
    plot_obj.outline_line_color = "white"
    plot_obj.xgrid.grid_line_color = None
    plot_obj.ygrid.grid_line_color = None
    plot_obj.xaxis.visible = False
    plot_obj.yaxis.visible = False
    plot_obj.add_layout(
        Label(x=4, y=3, text='REDOX', x_units='screen', y_units='screen', text_font_style="bold", text_font_size="8pt"),
        "left")
    plot_obj.line([-1, request.session["len"] + 1], [0.5, 0.5], color="black")
    x, y, identifier, width, color, range_nfo, type_nfo = [], [], [], [], [], [], []
    # patch_loc must exist in case of redox option
    for start, end in request.session['redox_regions'].items():
        start = start + 1
        color.append("#AA4499")
        x.append(((start + end) / 2))
        y.append(0.5)
        width.append(start - end)
        range_nfo.append("{}-{}".format(start, end))
    source = ColumnDataSource(data=dict(x=x, y=y, width=width, color=color, range_nfo=range_nfo))
    plot_obj.rect(x='x', y='y', width='width', height=0.65, name="boxes",
                  color='color', line_color="black", source=source,
                  nonselection_fill_alpha=1, nonselection_fill_color='color', nonselection_line_color="black",
                  nonselection_line_alpha=1)
    hover = HoverTool(names=["boxes"], point_policy='follow_mouse',
                      tooltips=[("Redox sensitive disordered region", "@range_nfo")])
    plot_obj.add_tools(hover)
    return plot_obj


def dl_mail_sender(request):
    email_data = {
        'title': request.POST.get('title'),
        'first_name': request.POST.get('first_name'),
        'last_name': request.POST.get('last_name'),
        'email_addr': request.POST.get('email'),
        'affiliation': request.POST.get('affiliation'),
        'liscence': request.POST.get('license'),
        'academic': request.POST.get('academic'),
        'ip': get_client_ip(request)}
    params = {
        'secret': '6LdPuEAUAAAAAFbqoMMs_MzOlqVqvFGqBEYIfMPE',
        'response': request.POST.get('g-recaptcha-response'),
        'remoteip': get_client_ip(request)
    }
    verify_rs = requests.get("https://www.google.com/recaptcha/api/siteverify", params=params, verify=True)
    verify_rs = verify_rs.json()
    if not verify_rs.get("success", False):
        error_message = "Invalid CAPTCHA"
        return render(request, "download.html", locals())

    if not all(i for i in email_data.values()):
        error_message = "You must fill all the fields!"
        return render(request, "download.html", locals())

    thr = threading.Thread(target=download_email_sender.send_iupred2a,
                           args=(email_data['email_addr'], email_data['title'], email_data['first_name'],
                                 email_data['last_name']))
    thr.setDaemon(True)
    thr.start()

    # Send the review emails
    thr = threading.Thread(target=download_email_sender.send_review,
                           args=(email_data,))
    thr.setDaemon(True)
    thr.start()

    with open("{}/logs/sended_to".format(DATA_DIR), "a") as fn:
        fn.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t\n".format(email_data['title'], email_data['first_name'],
                                                email_data['last_name'], email_data['email_addr'],
                                                email_data['affiliation'],
                                                get_client_ip(request)))
    return render(request, "download.html", {
        "error_message": "Thank you for downloading IUPred! The package has been successfully sent to {}.".format(
            email_data['email_addr'])})


def get_client_ip(request):
    """
    Gets the client IP address
    :param request:
    :return:
    """
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def get_ip_location(ip):
    url = 'http://ipinfo.io/{}/json'.format(ip)
    response = requests.get(url)
    try:
        return response.json()['loc'].split(',')
    except KeyError:
        return


def iupred_glob_text(_file):
    """
    Generates IUPred glob mode text
    :param _file:
    :return:
    """
    os.environ["IUPred_PATH"] = "{}/bin/iupred".format(DATA_DIR)
    glob_text = ""
    proc = subprocess.Popen('{}/bin/iupred/iupred_new {} {}'.format(DATA_DIR, _file, "glob"), shell=True,
                            stdout=subprocess.PIPE)
    res, err = proc.communicate()
    if err:
        exit(err)
    for line in res.decode().splitlines():
        if line.startswith("#"):
            continue
        glob_text += line + "\n"
    return glob_text


def pfam(_file):
    """
    Generates PFAM output
    :param _file: File to run PFAM for
    :return: List of dictionary for each PFAM anotation
    """
    os.environ["PERL5LIB"] = "{}/bin/pfamscan".format(DATA_DIR)
    proc = subprocess.Popen(
        'perl {0}/bin/pfamscan/pfam_scan.pl -dir {0}/bin/pfamscan/ -fasta {1}'.format(DATA_DIR, _file),
        shell=True,
        stdout=subprocess.PIPE)
    res, err = proc.communicate()
    diso_domains = []
    with open('{}/data/Pfam_disorder/disordered_Pfam_objects_with_support-manually_curated.txt'.format(DATA_DIR)) as fn:
        for line in fn:
            diso_domains.append(line.split()[1])
    if err:
        exit(err)
    _pfam = []
    for line in res.decode().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        _pfam.append({"name": line.split()[6], "type": line.split()[7], "real_start": int(line.split()[3]),
                      "real_end": int(line.split()[4])})
    for elem in _pfam:
        if elem['name'] in diso_domains:
            elem['type'] = "Disordered"

    return _pfam


def ptm(request):
    """
    Collect Phosphosite annotation
    :param request: Accession
    :return: Tuple of dictionaries for phosphorilation, acetilation, metilation
    """
    if not request.session["accession"]:
        return [], [], []

    phosphosite_annotation = ([], [], [])
    # All files are in the exact same format
    files = ['Phosphorylation_site_dataset', 'Acetylation_site_dataset', 'Methylation_site_dataset']
    for num, phos_site_file in enumerate(files):
        proc = subprocess.Popen(
            'fgrep {} {}/data/PhosphoSitePlus/{}'.format(request.session["accession"], DATA_DIR, phos_site_file),
            shell=True, stdout=subprocess.PIPE)
        res, err = proc.communicate()
        if err:
            exit(err)
        for line in res.decode().splitlines():
            if line.split()[2] == request.session["accession"] and line.split("\t")[10]:
                phosphosite_annotation[num].append(
                    {"pos": re.search(r'\d+', line.split("\t")[4]).group(), "res": line.split("\t")[4][0]})
    return phosphosite_annotation


def get_redox_regions(redox_values, iupred_values):
    """
    Calculate the redox sensitive regions
    :param redox_values: Redox Y coordinates
    :param iupred_values: IUPred Y coordiantes
    :return:
    """
    patch_loc = {}
    trigger = False
    opening_pos = []
    start, end = 0, 0
    counter = 0
    # Calculate possible position
    for idx, redox_val in enumerate(redox_values):
        if redox_val > 0.5 > iupred_values[idx] and redox_val - iupred_values[idx] > 0.3:
            opening_pos.append(idx)
    # Filter out where not enough possible position is found
    # Enlarge region where enough position if found
    for idx, redox_val in enumerate(redox_values):
        if redox_val - iupred_values[idx] > 0.15:
            if not trigger:
                start = idx
                trigger = True
            if idx in opening_pos:
                counter += 1
            end = idx
        else:
            trigger = False
            if end - start > 10 and counter:
                patch_loc[start] = end
            counter = 0
    if end - start > 10 and counter:
        patch_loc[start] = end
    # Combine close regions
    deletable = []
    for start, end in patch_loc.items():
        for start2, end2 in patch_loc.items():
            if start != start2 and start2 - end < 10 and start2 > start:
                patch_loc[start] = end2
                deletable.append(start2)
    for start in deletable:
        del patch_loc[start]
    return patch_loc


def experimental_disorder(request):
    """
    Calculates the experimental disorder for a given accession
    :return: List of tuples for each region
    """
    files = [('DIBS_entries_with_PDB_UniProt_UniRef90.txt', "DIBS", 5),
             ('DisProt_regions_mapped_to_UniProt_and_UniRef90-manually_curated.txt', 'DisProt', 4),
             ('MFIB_entries_with_PDB_UniProt_UniRef90.txt', "MFIB", 5)]
    result_lst = []
    if not request.session["accession"]:
        return result_lst
    for file_loc, ident, place_loc in files:
        proc = subprocess.Popen(
            'fgrep {} {}/data/{}'.format(request.session["accession"], DATA_DIR, file_loc), shell=True,
            stdout=subprocess.PIPE)
        res, err = proc.communicate()
        if err:
            exit(err)
        for line in res.decode().splitlines():
            result_lst.append({"start": int(line.split("\t")[place_loc].split("-")[0]),
                               "end": int(line.split("\t")[place_loc].split("-")[1]), "type": ident,
                               'id': line.split("\t")[0]})

    return result_lst


def pdb(accession):
    """
    Read the UNIPROT Accession API or EMBI
    :param accession:
    :return: Dictionary of PDBs in respect to overlap
    """
    pdb_map = {}
    try:
        url = requests.get("https://www.ebi.ac.uk/pdbe/api/mappings/{}".format(accession))
        data = json.loads(url.text)
        for i, j in data[accession]["PDB"].items():
            for q in j:
                if q["unp_end"] - q["unp_start"] < 2:
                    continue
                if i in pdb_map:
                    pdb_map[i].append({"start": q["unp_start"], "end": q["unp_end"]})
                else:
                    pdb_map[i] = [{"start": q["unp_start"], "end": q["unp_end"]}]
    except HTTPError:
        return pdb_map
    # Combine the overlapping segments
    res_map = {}
    for i, z in pdb_map.items():
        for j in z:
            id_set = set([])
            for u, k in pdb_map.items():
                for v in k:
                    if j["start"] == v["start"] and j["end"] == v["end"]:
                        id_set.add(u)
            res_map["{}_{}".format(j["start"], j["end"])] = {"start": j["start"], "end": j["end"], "ids": id_set}
    return res_map
