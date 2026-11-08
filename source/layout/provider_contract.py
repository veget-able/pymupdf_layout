"""Explicit CNN layout material adapter over the unchanged public model.

Native providers use their original path. No RF IDs, hidden vectors or model
probabilities are invented here. The provider owns its temporary model views;
the original model, sessions and extractor are never mutated.
"""
from collections import Counter
from copy import copy
import hashlib
import numpy as np


def array_identity(value):
    value=np.asarray(value)
    return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes()).hexdigest())


class _ObservedSession:
    def __init__(self,session,provider,name):
        self.session,self.provider,self.name=session,provider,name
    def __getattr__(self,name):
        return getattr(self.session,name)
    def run(self,names,inputs,*args,**kwargs):
        outputs=self.session.run(names,inputs,*args,**kwargs)
        self.provider.calls[self.name]+=1
        self.provider.feed_events.append(dict(model=self.name,inputs={k:array_identity(v) for k,v in inputs.items()}))
        return outputs


class CNNLayoutProvider:
    kind='cnn4gnn'
    expert=False
    def __init__(self,model):
        self._source_model=model
        self.calls=Counter()
        self.feed_events=[]
        self.errors=[]
        self.raw_events=[]
        self.runtime_events=[]
        self._data_owners={}
        self._identity=self.model_state()
        self._baseline=self._view()

    @property
    def model(self):
        return self._source_model

    def model_state(self):
        model=self.model;grid=model.table_grid_extractor
        return (id(model),id(model.session_s1),id(model.session_s2),id(grid._grid_sess),
                model.edge_threshold,model.det_score_thresh,model.det_nms_iou,
                grid.h_on_threshold,grid.v_on_threshold,grid.nms_min_dist,
                grid.filter_empty_lines,grid.snap_to_bbox_gaps,self.expert)

    @staticmethod
    def data_state(data):
        arrays=tuple((key,tuple(np.asarray(data[key]).shape),str(np.asarray(data[key]).dtype),
                      hashlib.sha256(np.asarray(data[key]).tobytes()).hexdigest())
                     for key in ('image','model_image','bboxes','bboxes_px','bboxes_src') if data.get(key) is not None)
        return arrays,tuple(data['text']),tuple(data['page_size']),tuple(data['source_size']),data['image_scale']

    @staticmethod
    def page_state(page):
        return (id(page.parent),page.number,page.xref,tuple(page.rect),
                tuple(page.cropbox),page.rotation,tuple(page.get_contents()))

    def reset_observations(self):
        self.calls.clear();self.feed_events.clear();self.errors.clear()
        self.raw_events.clear();self.runtime_events.clear();self._data_owners.clear()

    def create_data(self,page,*,use_ocr=False,**kwargs):
        data=self.model.create_data(page,use_ocr=use_ocr,**kwargs)
        self._data_owners[id(data)]=(data,self.page_state(page),bool(use_ocr),self.data_state(data))
        self.calls['create_data']+=1
        return data

    def _view(self,runtime=None,retained_prediction=None):
        model=copy(self.model)
        model.session_s1=_ObservedSession(self.model.session_s1,self,'Stage1')
        model.session_s2=_ObservedSession(self.model.session_s2,self,'Stage2')
        extractor=copy(self.model.table_grid_extractor)
        extractor._grid_sess=_ObservedSession(self.model.table_grid_extractor._grid_sess,self,'V5')
        original=extractor.predict
        if runtime is not None:
            runtime.structure_pixel_space='BGR'
            def execute(*args,**kwargs):
                runtime.stats['tgif_calls']+=1
                return original(*args,**kwargs)
            runtime.structure_predict=execute
            def retain(*args,**kwargs):
                runtime.stats['tgif_requested']+=1
                return retained_prediction(execute,args,kwargs),[]
            extractor.predict=retain
        model.table_grid_extractor=extractor
        return _CNNView(self,model,runtime)

    def bind_runtime(self,runtime,retained_prediction):
        return self._view(runtime,retained_prediction)

    def predict(self,page,**kwargs):
        return self._baseline.predict(page,**kwargs)

    def record_error(self,page,exc):
        record=dict(page=page.number,condition='HTML_TABLE_ENGINE_FAILED',
                    type=type(exc).__name__,message=str(exc))
        # Read only existing error-time producer caches, never extract again.
        words=getattr(page,'_refine_words_cache',None)
        blocks=getattr(page,'_table_raw_blocks',None)
        from pymupdf import table
        record['existing_material']={
            'words':list(words) if words is not None else None,
            'raw_text_blocks':[b for b in (blocks or ()) if b.get('type')==0],
            'characters':list(table.CHARS),
            'control_glyph_records':getattr(page,'_table_control_glyphs',None),
            'existing_font_symbol_maps':getattr(page.parent,'_table_font_symbol_maps',None),
            'source_material':getattr(exc,'source_material',None),
        }
        self.errors.append(record)


class _CNNView:
    def __init__(self,provider,model,runtime):
        self.provider,self.model,self.runtime=provider,model,runtime
    def predict(self,page,**kwargs):
        provider=self.provider
        if provider.model_state()!=provider._identity:
            raise RuntimeError('CNN provider model/session/options identity changed within frozen scope')
        if kwargs.get('logits_hook') is not None:
            raise ValueError('Frozen CNN provider has no LocalExpert/logits replacement')
        wanted_raw=bool(kwargs.pop('return_raw',False))
        kwargs.pop('return_logits',None)
        data=kwargs.get('data_dict')
        use_ocr=bool(kwargs.get('use_ocr',False))
        if data is None:
            data=provider.create_data(page,use_ocr=use_ocr)
            kwargs['data_dict']=data
        else:
            owner=provider._data_owners.get(id(data))
            if owner is None or owner[0] is not data or owner[1:]!=(provider.page_state(page),use_ocr,provider.data_state(data)):
                raise ValueError('CNN prepared material belongs to a different page/geometry/OCR state')
        raw=self.model.predict(page,return_raw=True,return_logits=True,**kwargs)
        # Capture the actual public final logits, after any public model hook.
        if self.runtime is not None and raw.get('edge_logits') is not None:
            self.runtime.stats['gnn_calls']+=1
        provider.raw_events.append(dict(page=page.number,state=provider.page_state(page),raw=raw,texts=tuple(data['text'])))
        if wanted_raw:return raw['groups']
        return raw['det_result']
