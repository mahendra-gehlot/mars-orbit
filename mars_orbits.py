import pandas as pd
import numpy as np
from numpy import pi as pi
import datetime
import matplotlib.pyplot as plt
import scipy.optimize
import seaborn as sns
from scipy.optimize import minimize


class PrepareDF:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(self.path)
        self.time_duration = list()
        self.longitudes = list()

    def prepare(self):
        for ind in range(1, len(self.df) + 1):
            time1 = datetime.datetime(self.df['Year'][ind - 1],
                                      self.df['Month'][ind - 1],
                                      self.df['Day'][ind - 1],
                                      self.df['Hour'][ind - 1],
                                      self.df['Minute'][ind - 1])
            time0 = datetime.datetime(self.df['Year'][0], self.df['Month'][0],
                                      self.df['Day'][0], self.df['Hour'][0],
                                      self.df['Minute'][0])
            self.time_duration.append(
                (time1 - time0).total_seconds() / (3600 * 24))

        self.df['time_in_days'] = self.time_duration

        for ind in range(len(self.df)):
            zodiac = self.df['ZodiacIndex'][ind]
            degree = self.df['Degree'][ind]
            minute = self.df['Minute.1'][ind]
            seconds = self.df['Second'][ind]
            angle = (30 * zodiac) + degree + (minute / 60) + (seconds / 3600)
            self.longitudes.append(angle)

        self.df['longitude_degree'] = self.longitudes
        self.df = self.df[['time_in_days', 'longitude_degree']]

        return np.array(self.df['longitude_degree']), np.array(
            self.df['time_in_days'])


mars_data = PrepareDF('data/01_data_mars_opposition_updated.csv')
longitude_series, time_series = mars_data.prepare()


class MarsModel:
    def __init__(self,
                 c,
                 r,
                 e1,
                 e2,
                 z,
                 s,
                 times=time_series,
                 oppositions=longitude_series):
        self.cx = np.cos(c)
        self.cy = np.sin(c)
        self.e1 = e1
        self.ex = e1 * np.cos(e2)
        self.ey = e1 * np.sin(e2)
        self.z1 = (s * times + z) % (2 * np.pi)
        self.r = r
        self.oppositions = np.radians(oppositions)
        self.err_cal = list()

    def calculate_error(self):
        const = self.ey - self.ex * np.tan(self.z1) - self.cy
        a = 1 + np.square(np.tan(self.z1))
        b = (2 * np.tan(self.z1) * const - 2 * self.cx)
        c = np.square(const) + np.square(self.cx) - np.square(self.r)

        d = np.sqrt(b**2 - 4 * a * c)

        x1 = (-b + d) / (2 * a)
        x2 = (-b - d) / (2 * a)

        y1 = self.ey + (x1 - self.ex) * np.tan(self.z1)
        y2 = self.ey + (x2 - self.ex) * np.tan(self.z1)

        errs = []

        for ind in range(len(self.oppositions)):
            x1_i = x1[ind]
            x2_i = x2[ind]
            y1_i = y1[ind]
            y2_i = y2[ind]

            angle1 = np.arctan2(y1_i, x1_i)
            angle2 = np.arctan2(y2_i, x2_i)

            actual_ang = self.oppositions[ind]

            abs1 = abs(angle1 - actual_ang)
            abs2 = abs(angle2 - actual_ang)

            if abs1 > pi:
                abs1 = abs1 - 2 * pi
            if abs2 > pi:
                abs2 = abs2 - 2 * pi
            angles = np.array([abs1, abs2])
            min_ind = np.argmin(np.abs(angles))
            errs.append(angles[min_ind])

        self.err_cal = np.array(errs)

        return None

    def measured_err(self):
        if self.err_cal.any():
            return self.err_cal
        else:
            print('Fit Model First!')

    def fit(self):
        self.calculate_error()
        return max(np.abs(self.err_cal))

    def plot(self):
        x, y = 0, 0
        ax = plt.gca()
        ax = sns.lineplot(x=[x, self.ex], y=[y, self.ey])
        xs = 10 * np.cos(self.z1) + self.ex
        ys = 10 * np.sin(self.z1) + self.ey
        for i in range(np.size(self.z1)):
            ax = sns.lineplot(x=[self.ex, xs[i]],
                              y=[self.ey, ys[i]],
                              linestyle="dashed")
        # sun
        ax = sns.lineplot(x=[x, self.cx], y=[y, self.cy])
        ox = 10 * np.cos(self.oppositions)
        oy = 10 * np.sin(self.oppositions)
        for j in range(np.size(self.oppositions)):
            ax = sns.lineplot(x=[0, ox[j]], y=[0, oy[j]])
        circle = plt.Circle((self.cx, self.cy), self.r, color='b', fill=False)
        ax.add_patch(circle)
        ax.set_aspect("equal")

        plt.show()


class SingleOptimizer:
    def __init__(self, variable, var_ini, args, bound, model=MarsModel):
        self.bound = bound
        self.variable = variable
        self.variable_ini = var_ini
        self.args = args
        self.model = model

    def objective(self, value, name_):
        opti_vari = dict()
        opti_vari[name_] = value
        c = opti_vari['c'] if 'c' in opti_vari else self.args['c']
        r = opti_vari['r'] if 'r' in opti_vari else self.args['r']
        e1 = opti_vari['e1'] if 'e1' in opti_vari else self.args['e1']
        e2 = opti_vari['e2'] if 'e2' in opti_vari else self.args['e2']
        z = opti_vari['z'] if 'z' in opti_vari else self.args['z']
        s = opti_vari['s'] if 's' in opti_vari else self.args['s']

        return self.model(c, r, e1, e2, z, s).fit()

    def run(self):
        res = minimize(self.objective,
                       x0=np.array([self.variable_ini]),
                       args=self.variable,
                       method='Nelder-Mead',
                       bounds=[self.bound],
                       options={
                           'maxiter': 10000,
                           'adaptive': True,
                           'fatol': 1e-9
                       })
        return res.x


class InternalOptimizer:
    def __init__(self,
                 r,
                 s,
                 initialization,
                 times=time_series,
                 oppositions=longitude_series,
                 model=MarsModel):
        self.r = r
        self.s = s
        self.times = times
        self.oppositions = oppositions
        self.model = model
        self.ini = initialization

    def objetive_fun(self, args):
        c = args[0]
        e1 = args[1]
        e2 = args[2]
        z = args[3]
        return self.model(c, self.r, e1, e2, z, self.s, self.times,
                          self.oppositions).fit()

    def run(self):
        res = minimize(self.objetive_fun,
                       x0=self.ini,
                       method='Nelder-Mead',
                       bounds=((0, None), (0, 0.8 * self.r), (0, None),
                               (0, None)),
                       options={
                           'maxiter': 1000,
                       })
        arg = res.x
        return arg

    def run_brute(self):
        r_ranges = (slice(-pi, pi, 0.2), slice(0, self.r, 1),
                    slice(-pi, pi, 0.2), slice(-pi, pi, 0.2))
        res_brute = scipy.optimize.brute(self.objetive_fun,
                                         ranges=r_ranges,
                                         finish=scipy.optimize.fmin)
        return res_brute, self.objetive_fun(res_brute)


def MarsEquantModel(c, r, e1, e2, z, s, times, oppositions):
    """
    The MarsEquant model as per Equant Hypothesis by Kepler
    All input angles are in radians
    :param c: angle of center of Mars measured from aries line
    :type c: float
    :param r: radius of Mars orbit
    :type r: float
    :param e1: radius of Equant
    :type e1: float
    :param e2: angle of Equant with aries line
    :type e2: float
    :param z: calculated angle of opposition from equant measured from aries line
    :type z: float
    :param s: angular speed in rad/day
    :type s: float
    :param times: time of oppositions in days (assuming first opposition happened on day 0)
    :type times: np.array
    :param oppositions: series of angle of oppositions in degree
    :type oppositions: np.array
    :return: numpy array of errors, max
    :rtype: numpy.array , float
    """
    model = MarsModel(c, r, e1, e2, z, s)
    max_error = model.fit()
    errors = model.measured_err()

    return errors, max_error


Question 01
print('QUESTION 01')
# errors, maxError = MarsEquantModel(2.5991373255251933, 8.18212891, 1.519648971343584,
#                     2.597157108784485, 0.9744796022943893, 0.00914694657579844,
#                     time_series, longitude_series)
errors, maxError = MarsEquantModel(2.6, 8.44, 1.5, 2.5, 0.97, 2 * pi / 687,
                                   time_series, longitude_series)
print(f'Errors: {errors}')
print(f'MaxError: {maxError}')

args = {
    'c': 2.6016086828346126,
    'e1': 1.6235464744913284,
    'e2': 2.6000569634327957,
    'z': 0.9745581981092616,
    'r': 8.732190304526338,
    's': 0.00914694657579844
}


def bestOrbitInnerParams(r,
                         s,
                         opti_args,
                         times=time_series,
                         oppositions=longitude_series):
    counter = 20
    # Initialization of variable
    opti_args['s'] = s
    opti_args['r'] = r

    def run_exhaustive(r_new, s_new):
        opti_args['r'] = r_new
        opti_args['s'] = s_new
        search_var = ['e1', 'e2', 'z', 'c']
        bounds = [(1, 0.8 * opti_args['r']), (0, 2 * pi), (0, 2 * pi),
                  (0, 2 * pi)]
        # Running single values optimizer 0: e1, 1: e2, 2:z, 3:c
        for i in 2 * [3, 2, 1, 0]:
            var = search_var[i]
            question2 = SingleOptimizer(variable=var,
                                        var_ini=opti_args[var],
                                        args=opti_args,
                                        bound=bounds[i],
                                        model=MarsModel)
            opti_args[var] = question2.run()[0]

        init = [
            opti_args['c'], opti_args['e1'], opti_args['e2'], opti_args['z']
        ]
        q2 = InternalOptimizer(r=opti_args['r'],
                               s=opti_args['s'],
                               initialization=init)
        opti_args['c'], opti_args['e1'], opti_args['e2'], opti_args[
            'z'] = q2.run()

    for temp_i in range(counter):
        run_exhaustive(r, s)

    q2_model = MarsModel(opti_args['c'], opti_args['r'], opti_args['e1'],
                         opti_args['e2'], opti_args['z'], opti_args['s'])
    max_error = q2_model.fit()
    q2_model.measured_err()

    return opti_args['c'], opti_args['e1'], opti_args['e2'], opti_args[
        'z'], q2_model.measured_err(), max_error


# Question 2
print('\nQUESTION 02')
R = 8.18212891
S = 0.00914694657579844
q2_args = dict()
q2_args['c'] = 2.6146184895358364
q2_args['e1'] = 1.5061522094620658
q2_args['e2'] = 2.605186417842634
q2_args['z'] = 0.9786185167839457
c, e1, e2, z, errors, maxError = bestOrbitInnerParams(
    r=R,
    s=S,
    opti_args=q2_args,
    times=time_series,
    oppositions=longitude_series)
print(f'c:{c}, e1:{e1}, e2:{e2}, z:{z}')
print(f'Errors: {errors}')
print(f'MaxError: {maxError}')

q3_args = dict()
q3_args['c'] = c
q3_args['e1'] = e1
q3_args['e2'] = e2
q3_args['z'] = z
q3_args['r'] = R
q3_args['s'] = S


# Question 3
def objective_3(s):
    c, e1, e2, z, errors, maxError = bestOrbitInnerParams(q3_args['r'],
                                                          s,
                                                          opti_args=q3_args)

    return maxError


def bestS(r, args=q3_args, times=time_series, oppositions=longitude_series):
    q3_args['r'] = r
    res = minimize(objective_3,
                   x0=np.array(q3_args['s']),
                   method='Nelder-Mead',
                   tol=1e-16,
                   bounds=[(0.97 * 0.00914694657579844,
                            1.03 * 0.00914694657579844)])
    q3_args['s'] = res.x[0]
    errs, max_err = MarsEquantModel(q3_args['c'], q3_args['r'], q3_args['e1'],
                                    q3_args['e2'], q3_args['z'], q3_args['s'],
                                    time_series, longitude_series)

    return q3_args['s'], errs, max_err


# question 03
print('\nQUESTION 03')
s, errors, maxError = bestS(R)
print(f'Best s:{s}')
print(f'Errors: {errors}')
print(f'MaxError: {maxError}')
print(q3_args)

q4_args = q3_args


def objective_4(r):
    c, e1, e2, z, errors, maxError = bestOrbitInnerParams(r,
                                                          q4_args['s'],
                                                          opti_args=q4_args)
    return maxError


def bestR(s, args=q4_args, times=time_series, oppositions=longitude_series):
    q4_args['s'] = s
    res = minimize(objective_4,
                   x0=np.array([q4_args['r']]),
                   method='Nelder-Mead',
                   tol=1e-16)
    q4_args['r'] = res.x[0]
    errs, max_err = MarsEquantModel(q4_args['c'], q4_args['r'], q4_args['e1'],
                                    q4_args['e2'], q4_args['z'], q4_args['s'],
                                    time_series, longitude_series)

    return q4_args['r'], errs, max_err


# question 04
print('\nQUESTION 04')
r, errors, maxError = bestR(q4_args['s'])
q4_args['r'] = r
print(f'Best r:{r}')
print(f'Errors: {errors}')
print(f'MaxError: {maxError}')
print(q4_args)
