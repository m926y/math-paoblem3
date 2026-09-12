import unittest
from dataclasses import replace
import numpy as np
from microgrid.config import Config
from microgrid.dispatch import solve,execute,ETA,EMIN,EMAX,QMAX
from microgrid.forecast import ForecastEngine,weighted_quantile
from microgrid.settlement import settle,revision_fee
from microgrid.timegrid import interval,validate_end_labels


class ModelTests(unittest.TestCase):
    def test_settlement_cases(self):
        for grid,want in ((80,90),(100,100),(120,130)):
            self.assertAlmostEqual(settle([1],[grid],[0],[100])['total_cost_cny'],want)

    def test_repeated_revision_and_cross_price_emergency(self):
        self.assertEqual(revision_fee([1],[100],[120])+revision_fee([1],[120],[100]),20)
        self.assertEqual(settle([1,2],[0,0],[10,20])['emergency_cost_cny'],250)
        self.assertEqual(settle([1],[100],[0],[100])['adjustment_cost_cny'],0)

    def test_time_grid_and_pulse(self):
        validate_end_labels(np.arange(1,145)/144)
        self.assertEqual(interval(0),'0:00-0:10')
        self.assertEqual(interval(60),'10:00-10:10')
        self.assertEqual(interval(143),'23:50-24:00')
        bad=np.arange(1,145)/144;bad[60]=bad[59]
        with self.assertRaises(ValueError):validate_end_labels(bad)
        pulse=np.zeros(144);pulse[24]=1
        self.assertEqual(pulse[:24].sum(),0);self.assertEqual(pulse[24:48].sum(),1)

    def test_joint_error_cancels(self):
        error=np.array([[10.,-20.],[-10.,20.]])
        np.testing.assert_equal(weighted_quantile(error-error,.8,[1,1]),[0,0])

    def test_extreme_execution(self):
        for net in (np.ones(144)*10000,-np.ones(144)*10000):
            x=execute(np.zeros(144),np.zeros(144),np.zeros(144),np.maximum(net,0),np.maximum(-net,0),6000)
            np.testing.assert_allclose(x['discharge']+x['emergency']-net-x['charge']-x['spill'],0,atol=1e-8)
            np.testing.assert_allclose(np.diff(x['energy']),ETA*x['charge']-x['discharge']/ETA,atol=1e-8)
            self.assertGreaterEqual(x['energy'].min(),EMIN)
            self.assertLessEqual(x['energy'].max(),EMAX)
            self.assertLessEqual(max(x['charge'].max(),x['discharge'].max()),QMAX)

    def test_scenario_price_tradeoff(self):
        # 电池空且单段同价：需求0/100等概率，短缺5倍，最优购买100。
        cfg=Config(mode='scenario',lexicographic=False)
        result=solve([1],[[0],[100]],[.5,.5],EMIN+1e-6,cfg)
        self.assertAlmostEqual(result['grid'][0],100,places=4)

    def test_lexicographic_no_overlap(self):
        x=solve(np.r_[np.ones(12)*.4,np.ones(12)*1.2],np.r_[np.ones(12)*-300,np.ones(12)*400][None,:],
                [1],6000,Config())
        self.assertLessEqual(np.minimum(x['charge'],x['discharge']).max(),1e-6)
        self.assertLessEqual(x['objective']-x['primary_objective'],2e-6)

    def test_cvar_reserve(self):
        cfg=Config(mode='scenario',cvar_weight=.1,reserve_factor=.2)
        x=solve(np.ones(12),np.array([np.ones(12)*50,np.ones(12)*300]),[.8,.2],6000,cfg,reserve_target=5000)
        self.assertTrue(np.isfinite(x['objective']))
        self.assertLess(x['constraint_residual'],1e-6)

    def test_causal_forecasts_and_cross_day(self):
        rng=np.random.default_rng(42)
        load=rng.uniform(3000,6000,(40,144));pv=rng.uniform(0,3000,(40,144))
        forecasts=rng.uniform(0,3000,(40,4,24));base=np.full(144,4000)
        for mode in ('weighted','ridge','adaptive'):
            cfg=Config(forecast=mode)
            original=ForecastEngine(load,pv,forecasts,base,base/2)
            changed_l=load.copy();changed_v=pv.copy();changed_f=forecasts.copy()
            changed_l[30,36:]+=10000;changed_l[31:]+=10000
            changed_v[30,36:]=0;changed_v[31:]=0
            changed_f[30,2:]+=9000;changed_f[31:]+=9000
            changed=ForecastEngine(changed_l,changed_v,changed_f,base,base/2)
            a=original.horizon(3,30,6,cfg);b=changed.horizon(3,30,6,cfg)
            for key in ('load','pv','errors','margin'):
                np.testing.assert_array_equal(a[key],b[key])
            # 整个0点购电优化也不得被当天未知实测改变。
            a=original.horizon(2,30,0,cfg);b=changed.horizon(2,30,0,cfg)
            np.testing.assert_array_equal(a['net'],b['net'])

    def test_execution_prefix_invariance(self):
        load=np.ones(144)*100;altered=load.copy();altered[60:]=10000
        args=(np.zeros(144),np.zeros(144),np.zeros(144))
        a=execute(*args,load,np.zeros(144),6000);b=execute(*args,altered,np.zeros(144),6000)
        for key in ('charge','discharge','emergency','energy'):
            np.testing.assert_equal(a[key][:60],b[key][:60])

    def test_published_plan_future_invariance(self):
        from microgrid.simulation import simulate
        rng=np.random.default_rng(19)
        load=rng.uniform(3000,5000,(3,144));pv=rng.uniform(0,1000,(3,144))
        f=np.full((3,4,24),500.)
        altered=load.copy();altered[1,36:]+=5000;altered[2:]+=5000
        engines=[ForecastEngine(x,pv,f,np.full(144,4000.),np.full(144,500.)) for x in (load,altered)]
        plans=[simulate(engine,3,np.full(144,.7),Config(forecast='weighted'),snapshots=True)[1] for engine in engines]
        for a,b in zip(*plans):
            if a['day']<1 or a['day']==1 and a['issue']<=6:
                np.testing.assert_allclose(a['grid'],b['grid'],rtol=0,atol=1e-8)
                self.assertAlmostEqual(a['energy_kwh'],b['energy_kwh'],places=7)


if __name__=='__main__':unittest.main()
