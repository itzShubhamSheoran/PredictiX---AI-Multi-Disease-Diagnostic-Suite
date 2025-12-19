import jsonfile from "jsonfile";
import moment from "moment";
import simpleGit from "simple-git";
import random from "random";

const git = simpleGit();
const path = "./data.json";


const TOTAL_WEEKS = 52;
const DAYS_IN_WEEK = 7;
const TOTAL_CELLS = TOTAL_WEEKS * DAYS_IN_WEEK;

const FILL_PERCENTAGE = 0.15;
const TARGET_DAYS = Math.floor(TOTAL_CELLS * FILL_PERCENTAGE); 

// Track already-used cells
const usedDays = new Set();


const getRandomDay = () => {
  let x, y, key;

  do {
    x = random.int(0, 51); // week
    y = random.int(0, 6);  // day (Sun–Sat)
    key = `${x}-${y}`;
  } while (usedDays.has(key));

  usedDays.add(key);
  return { x, y };
};


const makeCommit = (x, y) => {
  const date = moment()
    .subtract(1, "y")
    .add(1, "d")
    .add(x, "w")
    .add(y, "d")
    .hour(random.int(9, 20))   
    .minute(random.int(0, 59))
    .format();

  const data = { date };
  jsonfile.writeFileSync(path, data);

  git.add([path]).commit("chore: update", { "--date": date });
};

const make15PercentCommits = async (count) => {
  if (count === 0) {
    console.log("✅ 15% commits created. Pushing to GitHub...");
    await git.push();
    return;
  }

  const { x, y } = getRandomDay();

  const commitsPerDay =
    y === 0 || y === 6 ? random.int(1, 2) : random.int(1, 3);

  for (let i = 0; i < commitsPerDay; i++) {
    makeCommit(x, y);
  }

  make15PercentCommits(count - 1);
};

make15PercentCommits(TARGET_DAYS);
