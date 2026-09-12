import unittest
import numpy as np
from q4.model import PriceEngine,Settings,scenario_solve,simulate
from q4.audit import audit
from microgrid.forecast import ForecastEngine


class Q4Tests(unittest.TestCase):
    def test_absolute_cross_day_price(self):
        p=np.tile(np.arange(1,145,dtype=float),(3,1))
        p[1]+=1000
        engine=PriceEngine(p)
        h=engine.horizon(2,18,'persistence')
        # Bias is capped, but cannot import day 3 or wrap observed current-day early values.
        p2=p.copy();p2[2,108:]+=50000
        np.testing.assert_array_equal(h,PriceEngine(p2).horizon(2,18,'persistence'))
        base=engine.profile(2,3,'persistence')
        np.testing.assert_array_equal(base,p[1])

    def test_price_net_dependence_changes_order(self):
        cfg=Settings(variant='C0')
        joint=scenario_solve(np.array([[1.],[10.]]),np.array([[0.],[100.]]),[.9,.1],1200.000001,np.array([np.nan]),cfg)
        fixed=scenario_solve(np.array([[1.9],[1.9]]),np.array([[0.],[100.]]),[.9,.1],1200.000001,np.array([np.nan]),cfg)
        self.assertAlmostEqual(joint['grid'][0],100,places=4)
        self.assertAlmostEqual(fixed['grid'][0],0,places=4)

    def test_risk_and_adjustment(self):
        for risk in (0,.15):
            x=scenario_solve(np.array([[1.],[2.]]),np.array([[80.],[120.]]),[.5,.5],1200.000001,
                             np.array([100.]),Settings(variant='C15',risk_weight=risk))
            self.assertTrue(np.isfinite(x['objective']))
            self.assertLess(x['constraint_residual'],1e-6)

    def test_causal_full_plans_and_physics(self):
        rng=np.random.default_rng(20260913)
        load=rng.uniform(3000,5000,(4,144));pv=rng.uniform(0,1000,(4,144));f=np.full((4,4,24),500.)
        price=rng.uniform(.3,1.4,(4,144))
        altered_l=load.copy();altered_l[2,36:]+=5000;altered_l[3:]+=5000
        altered_p=price.copy();altered_p[2,36:]+=3;altered_p[3:]+=3
        for variant in ('B','C0'):
            outputs=[]
            for l,p in ((load,price),(altered_l,altered_p)):
                engine=ForecastEngine(l,pv,f,np.full(144,4000.),np.full(144,500.))
                records,plans,s=simulate(engine,PriceEngine(p),3,Settings(variant=variant),snapshots=True)
                audit(records,s,3,plans);outputs.append(plans)
            for a,b in zip(*outputs):
                if a['day']<2 or a['day']==2 and a['issue']<=6:
                    np.testing.assert_allclose(a['grid'],b['grid'],rtol=0,atol=1e-7)
                    np.testing.assert_allclose(a['price_prediction'],b['price_prediction'],rtol=0,atol=1e-9)

    def test_q2_freeze(self):
        l=np.full((3,144),4000.);v=np.zeros_like(l);p=np.tile(np.linspace(.3,1.4,144),(3,1))
        engine=ForecastEngine(l,v,np.zeros((3,4,24)),l[0],v[0])
        rec,plans,s=simulate(engine,PriceEngine(p),2,Settings(),snapshots=True)
        self.assertTrue(audit(rec,s,2,plans)['passed'])


if __name__=='__main__':unittest.main()
