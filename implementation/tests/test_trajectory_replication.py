import unittest
from unittest import mock
from implementation.src.contracts import ContractViolation
from implementation.src.pilot_execution_freeze import TRAJECTORY_REVISION
from implementation.src.pilot_runtime import PilotRuntime, phase_order


class ReplicationTests(unittest.TestCase):
    def test_only_explicit_revision_changes_outcome_gate_policy(self):
        r=PilotRuntime.__new__(PilotRuntime)
        r.freeze={'exploratory_revision':TRAJECTORY_REVISION,'execution':{'roots':[4102,4103,4104]}}
        r._root_gates=mock.Mock(side_effect=AssertionError('not an entry gate in replication'))
        r._training_gates(4102,'INIT')
        for root,phase in [(4101,'INIT'),(4102,'H_A'),(4104,'H_BASE')]:
            with self.assertRaisesRegex(ContractViolation,'SCOPE_VIOLATION'):r._training_gates(root,phase)
        r.freeze={}
        r._root_gates=mock.Mock(return_value={'measurement':{'status':'BLOCKED_MEASUREMENT'}})
        with self.assertRaisesRegex(ContractViolation,'BLOCKED_TRAINING_GATES'):r._training_gates(4102,'INIT')

    def test_replication_has_full_independent_initialization_and_no_history(self):
        order=phase_order({'roots':[4102,4103,4104],'start_phase':'INIT','history_enabled':False})
        self.assertEqual(len(order),18)
        for root in (4102,4103,4104):
            self.assertEqual([p for r,p in order if r==root],['INIT','CORPUS','T0','T1','T2','T3'])


if __name__=='__main__':unittest.main()
